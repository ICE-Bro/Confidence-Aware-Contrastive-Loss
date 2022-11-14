# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead
from mmcv.runner import force_fp32
from ..losses import accuracy
from .ocr_head import OCRHead

class Res(nn.Module):
    def __init__(self, module):
        super(Res, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class CACOCRHead(BaseCascadeDecodeHead):

    def __init__(self, ocr_channels, scale=1, proj_channel=256,
                 weight=1.0, **kwargs):
        super(CACOCRHead, self).__init__(**kwargs)
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
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

        self.weight_branch = OCRHead(
            in_channels=kwargs.get('in_channels', None),
            in_index=self.in_index,
            input_transform='resize_concat',
            channels=self.channels,
            ocr_channels=self.ocr_channels,
            dropout_ratio=self.dropout_ratio,
            num_classes=self.num_classes,
            norm_cfg=self.norm_cfg,
            align_corners=self.align_corners,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=weight)
        )
        self.weight = weight


    def forward(self, inputs, prev_output, train=True):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)
        if train:
            proj_feature = self.proj(x)
            weight_logit = self.weight_branch([inputs[i].detach() for i in self.in_index], prev_output.detach())
            return output, proj_feature, weight_logit
        return output

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        seg_logit, proj_feature, weight_logit = self.forward(inputs, prev_output)
        losses = self.losses(seg_logit, proj_feature, weight_logit, gt_semantic_seg, img_metas=img_metas)
        return losses

    def forward_test(self, inputs, prev_output, img_metas, test_cfg, **kwargs):
        seg_logit = self.forward(inputs, prev_output, train=False)
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



