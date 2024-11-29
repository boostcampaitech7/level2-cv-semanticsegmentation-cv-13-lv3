dataset_type = 'XRayDataset'

train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=(1024, 1024)),
            dict(type='TransposeAnnotations'),
        # CLAHE 변환 (Albumentations 활용)
            dict(type='CLAHE', clip_limit=4.0, tile_grid_size=(4, 4)),
            dict(type='PackSegInputs')
        ]

valid_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=(1024, 1024)),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]

test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1024, 1024)),
            dict(type='PackSegInputs')
        ]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        image_files=[],
        label_files=[],
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        image_files=[],
        label_files=[],
        pipeline=valid_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        image_files=[],
        label_files=None,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='DiceMetric')
val_cfg = dict(type='ValLoop')

test_evaluator = dict(type='RLEncoding')
test_cfg = dict(type='TestLoop')
