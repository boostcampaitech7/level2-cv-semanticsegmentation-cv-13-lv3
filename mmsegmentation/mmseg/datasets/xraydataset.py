import copy
import numpy as np
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmengine.dataset import Compose

from .basesegdataset import BaseSegDataset

from .xrayconstants import CLASSES, CLASS2IND, IND2CLASS, PALETTE

import json
import cv2

@DATASETS.register_module()
class XRayDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
                'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
                'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
                'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
                'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
                'Triquetrum', 'Pisiform', 'Radius', 'Ulna'),
        palette=PALETTE,
        image_size=(2048,2048))

    def __init__(self, image_files, label_files, **kwargs):
        self.image_files = image_files
        self.label_files = label_files  # Optional for test set without labels
        super().__init__(img_suffix='.png', seg_map_suffix='.json', **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        
        data_list = []
        for idx, (img) in enumerate(self.image_files):
            data_info = dict(img_path = img)
            if self.label_files is not None:
                data_info['seg_map_path'] = self.label_files[idx]
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        return data_list

from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class LoadXRayAnnotations(BaseTransform):

    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

    IND2CLASS = {v: k for k, v in CLASS2IND.items()}
    def __init__(self, input_size=(2048, 2048)):
            self.input_size = input_size

    def transform(self, result):
        label_path = result['seg_map_path']

        # process a label of shape (H, W, NC)
        label_shape = self.input_size + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations['annotations']
        
        # iterate each class
        for ann in annotations:
            c = ann['label']
            class_ind = CLASS2IND[c]
            points = np.array(ann['points'])
            
            # polygon to mask
            class_label = np.zeros(self.input_size, dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        result['gt_seg_map'] = label
        
        return result
    
@TRANSFORMS.register_module()
class TransposeAnnotations(BaseTransform):
    def transform(self, result):
        result['gt_seg_map'] = np.transpose(result['gt_seg_map'], (2, 0, 1))
        
        return result