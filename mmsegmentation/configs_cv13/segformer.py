_base_ = [
    '../configs/_base_/models/segformer_mit-b0.py',
    './schedule_50epoch.py',
    './dataset.py',
    './runtime.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
        type='SegDataPreProcessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        type='SegformerHead',
        num_classes=29,
        threshold=0.5,
        loss_decode=dict(
            use_sigmoid=True,
        )))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=5000,  # 총 에포크 수에 맞게 조정
        by_epoch=False)
]
