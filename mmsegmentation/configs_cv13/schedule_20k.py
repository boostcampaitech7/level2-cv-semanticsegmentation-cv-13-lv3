max_iters = 20000

# training schedule for 50 epochs
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1000)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=max_iters,  # 총 에포크 수에 맞게 조정
        by_epoch=False)
]
