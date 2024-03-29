_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                          class_weight=[0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
                                        1.0865, 1.0955, 1.0865, 1.1529, 1.0507])))

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '../data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CityscapesDataset',
        data_root='../data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type='CityscapesDataset',
        data_root='../data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='../data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))

# hook
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=12000)
evaluation = dict(interval=12000, metric='mIoU', pre_eval=True)
