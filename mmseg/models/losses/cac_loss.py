# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mmseg.ops import resize
from torch_scatter import scatter_mean
from ..builder import LOSSES

@LOSSES.register_module()
class CACLoss(nn.Module):
    def __init__(self,
                 max_sample=1024,
                 threshold=100,
                 loss_weight=0.1,
                 temperature=0.1,
                 base_temperature=0.07,
                 fp=False,
                 detach=False,
                 hard_sort_hardmining=False,
                 easy_sort_hardmining=False,
                 ignore_index=255,
                 loss_name='loss_cac'):
        super(CACLoss, self).__init__()
        self.max_sample = max_sample
        self.threshold = threshold
        self.loss_weight = loss_weight
        self.base_loss_weight = loss_weight
        self._loss_name = loss_name
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.base_temperature = base_temperature
        self.fp = fp
        self.hard_sort_hardmining = hard_sort_hardmining
        self.easy_sort_hardmining = easy_sort_hardmining
        self.detach = detach

    def _hard_sample_mining(self, predict, label, feature_map, probability, weight_logit):
        B = feature_map.shape[0]
        hard_sample_classes = []
        total_classes = 0

        for i in range(B):
            pixels_label_per = label[i]
            superpixels_class = torch.unique(pixels_label_per)
            class_per = [x for x in superpixels_class if x != self.ignore_index]
            class_per = [x for x in class_per if (pixels_label_per == x).nonzero().shape[0] > self.threshold]

            hard_sample_classes.append(class_per)
            total_classes += len(class_per)

        if total_classes == 0:
            return None, None, None

        feature_list = []
        label_list = []
        probability_list = []

        n_view = self.max_sample // total_classes
        n_view = min(n_view, self.threshold)

        for i in range(B):
            pixels_label_per = label[i]
            pixels_predict_per = predict[i]
            pixels_feature_per = feature_map[i]
            probability_per = probability[i]
            weight_logit_per = weight_logit[i]
            hard_sample_classes_per = hard_sample_classes[i]

            for cls_id in hard_sample_classes_per:
                if self.fp:
                    hard_indices = ((pixels_label_per != cls_id) & (pixels_predict_per == cls_id)).nonzero().squeeze(1)
                else:
                    hard_indices = ((pixels_label_per == cls_id) & (pixels_predict_per != cls_id)).nonzero().squeeze(1)

                easy_indices = ((pixels_label_per == cls_id) & (pixels_predict_per == cls_id)).nonzero().squeeze(1)

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    num_hard_keep = num_hard
                    num_easy_keep = num_easy

                if self.hard_sort_hardmining:
                    cls_pixels_logits_per = probability_per[hard_indices, cls_id]
                    cls_pixels_logits_per = cls_pixels_logits_per.argsort(dim=0, descending=True)
                    hard_indices = hard_indices[cls_pixels_logits_per[:num_hard_keep]]
                else:
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]

                if self.easy_sort_hardmining:
                    cls_pixels_logits_per = probability_per[easy_indices, cls_id]
                    cls_pixels_logits_per = cls_pixels_logits_per.argsort(dim=0)
                    easy_indices = easy_indices[cls_pixels_logits_per[:num_easy_keep]]
                else:
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                feature_list.append(pixels_feature_per[indices, :])
                label_list.append(pixels_label_per[indices])
                probability_list.append(weight_logit_per[indices, :])

        selected_feature = torch.cat(feature_list, dim=0)
        selected_label = torch.cat(label_list, dim=0)
        selected_probability = torch.cat(probability_list, dim=0)

        return selected_feature, selected_label, selected_probability

    def _contrastive(self, feature, label, probability):

        probability_mask = torch.gather(1 - probability, dim=1, index=repeat(label, 'n -> h n', h=label.shape[0]))
        probability_mask = (probability_mask + probability_mask.T) / 2

        label_map = label.unsqueeze(dim=1)
        label_map = torch.eq(label_map, label_map.T)

        size = feature.shape[0]
        if self.detach:
            dot_contrast = torch.div(torch.einsum('ij,jk->ik', feature, feature.T.detach()), self.temperature)
        else:
            dot_contrast = torch.div(torch.einsum('ij,jk->ik', feature, feature.T), self.temperature)

        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        dot_contrast = dot_contrast - logits_max.detach()

        logits_mask = ~torch.eye(size, dtype=torch.bool, device=label_map.device)
        label_map = torch.mul(label_map, logits_mask)
        probability_mask = torch.mul(probability_mask, logits_mask)
        pos_weighted_mask = torch.mul(probability_mask, label_map)

        exp_logits = torch.exp(dot_contrast) * probability_mask

        log_prob = dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-16)

        mean_log_prob_pos = (pos_weighted_mask * log_prob).sum(1) / (label_map.sum(1) + 1e-16)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = scatter_mean(loss, dim=0, index=label)
        loss = loss.mean() * self.loss_weight

        return loss

    def forward(self,
                predict,
                feature_map,
                label,
                weight_logit,
                **kwargs):
        """Forward function."""
        label = resize(
            input=label.unsqueeze(dim=1).float(),
            size=feature_map.shape[2:],
            mode='nearest').squeeze(dim=1).long()

        label = rearrange(label, 'b h w -> b (h w)')
        predict = rearrange(predict, 'b c h w -> b (h w) c')
        probability = F.softmax(predict, dim=2)
        predict = torch.argmax(predict, dim=2)
        weight_logit = F.softmax(rearrange(weight_logit, 'b c h w -> b (h w) c'), dim=2)
        feature_map = F.normalize(rearrange(feature_map, 'b c h w -> b (h w) c'), dim=2)

        features, labels, probability = \
            self._hard_sample_mining(predict, label, feature_map, probability, weight_logit)
        if features is None:
            return feature_map.mean() * 0
        loss = self._contrastive(features, labels, probability)

        return loss

    def weight_warmup_start(self):
        self.loss_weight = 0

    def weight_warmup_end(self):
        self.loss_weight = self.base_loss_weight
        return self.loss_weight

    @property
    def loss_name(self):
        return self._loss_name


if __name__ == '__main__':
    Loss = CACLoss(hard_sort_hardmining=True, easy_sort_hardmining=True)
    predict = torch.rand((2, 19, 128, 128))
    label = torch.randint(0, 19, (2, 512, 512))
    feature_map = torch.rand((2, 256, 128, 128), requires_grad=True)
    weight_logit = torch.rand((2, 19, 128, 128), requires_grad=True)
    loss = Loss(predict, feature_map, label, weight_logit)
    loss.backward()
    print(loss)
