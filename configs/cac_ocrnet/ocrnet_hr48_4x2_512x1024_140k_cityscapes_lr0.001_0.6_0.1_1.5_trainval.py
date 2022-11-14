_base_ = [
    '../_base_/models/cac_ocrnet_hr18.py', '../_base_/default_runtime.py'
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
        img_dir=['leftImg8bit/train', 'leftImg8bit/val'],
        ann_dir=['gtFine/train', 'gtFine/val'],
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
        pipeline=test_pipeline)
)

# hook
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=1.1, min_lr=0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=140000)
checkpoint_config = dict(by_epoch=False, interval=14000)
evaluation = dict(interval=14000, metric='mIoU', pre_eval=True)