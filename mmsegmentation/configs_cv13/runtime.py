default_scope = 'mmseg'

load_from = None
resume = True

work_dir = './work_dirs/segformer_experiment'

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         project='test',
                         name='test',
                         config={}
                     ))
                ]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'DEBUG'
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook',
            by_epoch=False,
            ignore_last=False,  # Log even the last iteration
            reset_flag=False,  # Don't reset the iteration counter
            logger_name='default')  # Use the default logger
    ])

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25, log_metric_by_epoch=False),  # 에포크 기반으로 변경
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),  # 5 에포크마다 체크포인트 저장
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))