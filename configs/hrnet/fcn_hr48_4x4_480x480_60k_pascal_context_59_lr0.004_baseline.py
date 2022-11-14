_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/pascal_context_59.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_60k.py'
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
            extra=dict(
                stage2=dict(num_channels=(48, 96)),
                stage3=dict(num_channels=(48, 96, 192)),
                stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        num_classes=59,
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072)
    ),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(
                     custom_keys={
                         'head': dict(lr_mult=10.)
                     }))
lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)