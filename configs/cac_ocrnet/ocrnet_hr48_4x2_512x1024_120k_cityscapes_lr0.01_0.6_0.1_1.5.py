_base_ = [
    '../_base_/models/cac_ocrnet_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_120k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            in_index=(0, 1, 2, 3),
            input_transform='resize_concat',
            kernel_size=1,
            num_convs=1,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
                              class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                            1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                            1.0865, 1.0955, 1.0865, 1.1529, 1.0507])),
        dict(
            type='CACOCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            proj_channel=256,
            weight=1.5,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
            loss_decode=[dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                      class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                    1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                    1.0865, 1.0955, 1.0865, 1.1529, 1.0507]),
                dict(type='CACLoss', loss_weight=0.6, temperature=0.1,
                     hard_sort_hardmining=True, easy_sort_hardmining=True, detach=False,
                     max_sample=1024, threshold=100)])])


custom_hooks = [
    dict(type='loss_weight_warmup_hook', start_iter=5000, loss_name="loss_cac")
]

lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)