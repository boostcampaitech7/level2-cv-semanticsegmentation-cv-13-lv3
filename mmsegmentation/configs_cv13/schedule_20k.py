max_iters = 20000

# training schedule for 50 epochs
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1000)
# learning policy
# Learning Rate Scheduler
param_scheduler = [
    dict(
        type='LinearLR',  # Warmup 단계
        start_factor=1e-5 / 1.25e-4,  # 시작 학습률 비율
        by_epoch=False,
        begin=0,
        end=2000  # Warmup 기간 (2k iterations)
    ),
    dict(
        type='PolyLR',  # Main Scheduler
        power=0.9,
        eta_min=1e-6,
        by_epoch=False,
        begin=2000,
        end=20000
    )
]