# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmcv.runner import force_fp32
from ..losses import accuracy

class Res(nn.Module):
    def __init__(self, module):
        super(Res, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

@HEADS.register_module()
class CACSegformerHead(BaseDecodeHead):

    def __init__(self,
                 proj_channel=256,
                 weight=1.0,
                 interpolate_mode='bilinear',
                 **kwargs):
        super(CACSegformerHead, self).__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.proj_channel = proj_channel
        self.proj = nn.Sequential(
            ConvModule(
                sum(self.in_channels),
                self.proj_channel,
                kernel_size=1,
                padding=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            Res(ConvModule(
                self.proj_channel,
                self.proj_channel,
                kernel_size=1,
                padding=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None)))

        self.weight_branch = nn.ModuleList()
        for i in range(num_inputs):
            self.weight_branch.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.weight_branch.append(nn.Sequential(
            ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg),
            ConvModule(
                self.channels,
                self.num_classes,
                kernel_size=1,
                padding=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None)))
        self.weight = weight

    def forward(self, inputs, train=True):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        cat_outs = torch.cat(outs, dim=1)
        out = self.fusion_conv(cat_outs)
        out = self.cls_seg(out)
        if train:
            proj_inputs = [inputs[i] for i in self.in_index]
            proj_upsampled_inputs = [
                resize(
                    input=x,
                    size=proj_inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in proj_inputs
            ]
            proj_upsampled_inputs = torch.cat(proj_upsampled_inputs, dim=1)
            proj_feature = self.proj(proj_upsampled_inputs)
            weight_outs = []
            for idx in range(len(inputs)):
                x = inputs[idx]
                conv = self.weight_branch[idx]
                weight_outs.append(
                    resize(
                        input=conv(x.detach()),
                        size=inputs[0].shape[2:],
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners))
            weight_out = torch.cat(weight_outs, dim=1)
            weight_logit = self.weight_branch[-1](weight_out)
            return out, proj_feature, weight_logit
        return out

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, proj_feature, weight_logit = self.forward(inputs)
        losses = self.losses(seg_logit, proj_feature, weight_logit, gt_semantic_seg, img_metas=img_metas)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **kwargs):
        seg_logit = self.forward(inputs, train=False)
        return seg_logit

    @force_fp32(apply_to=('seg_logit', 'proj_feature', 'weight_logit',))
    def losses(self, seg_logit, proj_feature, weight_logit, seg_label, **kwargs):
        """Compute segmentation loss."""
        loss = dict()
        cls_score = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        weight_score = resize(
            input=weight_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(cls_score, seg_label)
        else:
            seg_weight = None

        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            if loss_decode.loss_name == 'loss_lac':
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        predict=seg_logit,
                        feature_map=proj_feature,
                        label=seg_label,
                        weight_logit=weight_logit,
                        ignore_index=self.ignore_index,
                        **kwargs)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        predict=seg_logit,
                        feature_map=proj_feature,
                        label=seg_label,
                        weight_logit=weight_logit,
                        ignore_index=self.ignore_index,
                        **kwargs)

            else:
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_decode(
                        cls_score,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name] += loss_decode(
                        cls_score,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

                if loss_decode.loss_name + '_weight' not in loss:
                    loss[loss_decode.loss_name + '_weight'] = self.weight * loss_decode(
                        weight_score,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                else:
                    loss[loss_decode.loss_name + '_weight'] += self.weight * loss_decode(
                        weight_score,
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            cls_score, seg_label, ignore_index=self.ignore_index)
        loss['acc_seg_weight'] = accuracy(
            weight_score, seg_label, ignore_index=self.ignore_index)

        return loss
