_base_ = [
    '../_base_/models/cac_fcn_hr18.py', '../_base_/datasets/coco-stuff10k.py',
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
        proj_channel=256,
        weight=3.0,
        num_classes=171,
        in_channels=[48, 96, 192, 384],
        channels=sum([48, 96, 192, 384]),
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='CACLoss', loss_weight=3.5, temperature=0.1,
                          hard_sort_hardmining=True, easy_sort_hardmining=True, detach=False,
                          max_sample=1024, threshold=100)]),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg=dict(
                     custom_keys={
                         'head': dict(lr_mult=10.)
                     }))
lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

