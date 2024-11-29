_base_ = [
    './segformer.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(
        type='SegformerHead',
        num_classes=29,
        in_channels=[64, 128, 320, 512],
        threshold=0.5,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5),
            dict(type='DiceLoss',loss_weight=0.5)]))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1.25e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
