max_iters = 20000

# training schedule for 50 epochs
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters)
# learning policy
param_scheduler = [
    dict(
        type='ReduceLROnPlateau',
        mode='max',
        factor=0.5,
        patience=3,
        threshold=0.01,
        threshold_mode='rel',
        cooldown=1,
        min_lr=1e-6,
        verbose=True
    )
]