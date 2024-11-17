# training schedule for 50 epochs
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

work_dir = './work_dirs/segformer_experiment'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5, log_metric_by_epoch=False),  # 에포크 기반으로 변경
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),  # 5 에포크마다 체크포인트 저장
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))