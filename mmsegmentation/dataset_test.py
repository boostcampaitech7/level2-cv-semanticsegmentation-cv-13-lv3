
from mmseg.datasets import XRayDataset
from mmengine.registry import init_default_scope
init_default_scope('mmseg')

data_root = '../data/train/'
data_prefix=dict(img_path='DCM', seg_map_path='outputs_json')
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(512, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

dataset = XRayDataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline)
print(dataset)