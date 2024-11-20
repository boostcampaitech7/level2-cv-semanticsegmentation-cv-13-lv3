_base_ = [
    '../configs/_base_/models/fcn_hr18.py', 
    './schedule_20k.py',
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
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=29, loss_decode=dict(
            use_sigmoid=True,)))

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