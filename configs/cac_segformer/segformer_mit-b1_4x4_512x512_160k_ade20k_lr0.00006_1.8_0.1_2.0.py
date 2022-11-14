_base_ = [
    '../_base_/models/cac_segformer_mit-b0.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150,
        proj_channel=256,
        weight=2.0,
        sampler=dict(type='OHEMPixelSampler', thresh=0.9, min_kept=131072),
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                     dict(type='CACLoss', loss_weight=1.8, temperature=0.1,
                          hard_sort_hardmining=True, easy_sort_hardmining=True, detach=False,
                          max_sample=1024, threshold=100)]))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

custom_hooks = [
    dict(type='loss_weight_warmup_hook', start_iter=5000, loss_name="loss_cac")
]

