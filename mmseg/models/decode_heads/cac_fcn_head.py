# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses import accuracy

class Res(nn.Module):
    def __init__(self, module):
        super(Res, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

@HEADS.register_module()
class CACFCNHead(BaseDecodeHead):
    def __init__(self,
                 proj_channel=256,
                 weight=1.0,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(CACFCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.proj_channel = proj_channel
        self.proj = nn.Sequential(
            ConvModule(
                self.in_channels,
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

        self.weight_branch = nn.Sequential(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                self.channels,
                self.num_classes,
                kernel_size=1,
                padding=0,
                dilation=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None))
        self.weight = weight

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats, x

    def forward(self, inputs, train=True):
        output, x = self._forward_feature(inputs)
        if train:
            proj_feature = self.proj(x)
            weight_logit = self.weight_branch(x.detach())
            seg_logit = self.cls_seg(output)
            return seg_logit, proj_feature, weight_logit
        else:
            return self.cls_seg(output)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, proj_feature, weight_logit = self.forward(inputs)
        losses = self.losses(seg_logit, proj_feature, weight_logit, gt_semantic_seg, img_metas=img_metas)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, **kwargs):
        seg_logit = self.forward(inputs, train=False)
        return seg_logit

    @force_fp32(apply_to=('seg_logit', 'proj_feature', 'weight_logit', ))
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
            if loss_decode.loss_name == 'loss_cac':
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
