default_scope = 'mmseg'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

vis_backends = [dict(type='LocalVisBackend'),
                #dict(type='WandbVisBackend')
                ]

visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(by_epoch=False)
log_level = 'INFO'
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook',
            by_epoch=False,
            interval=1,  # Log after every iteration
            ignore_last=False,  # Log even the last iteration
            reset_flag=False,  # Don't reset the iteration counter
            logger_name='default')  # Use the default logger
        # dict(
        #     type='MMSegWandbHook',
        #     by_epoch=False,
        #     interval=1,
        #     with_step=False,
        #     init_kwargs=dict(
        #         entity='frostings',
        #         project='mmsegtest',
        #         name='mmsegtest'),
        #     log_checkpoint=True,
        #     log_checkpoint_metadata=True,
        #     num_eval_images=10)
    ])