import os
import numpy as np

import pandas as pd

from torch import Tensor
from typing import Any, Sequence
from mmseg.registry import METRICS
from mmengine.evaluator import BaseMetric 
from mmengine.structures import BaseDataElement

import torch
import torch.nn.functional as F

def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data
  

@METRICS.register_module()
class RLEncoding(BaseMetric):

    CLASSES = [
        'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
        'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
        'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
        'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
        'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
        'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
    ]

    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    def __init__(self,
                 threshold=0.5,
                 collect_device='cpu',
                 prefix=None,
                 **kwargs):
        
        self.rles = []
        self.filenames = []
        self.classes = []
        self.threshold = threshold

        super().__init__(collect_device=collect_device, prefix=prefix)

    @staticmethod
    def _encode_mask_to_rle(mask):
        '''
        mask: numpy array binary mask 
        1 - mask 
        0 - background
        Returns encoded run length 
        '''
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)# RLE로 인코딩된 결과를 mask map으로 복원합니다.

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:

        # 1. 메타데이터 추출
        image_names = [os.path.basename(sample['img_path']) for sample in data_samples]

        # 2. 모든 예측 데이터(`pred_sem_seg`)를 추출하여 배치로 결합
        pred_masks = torch.stack([data_sample['pred_sem_seg']['data'] for data_sample in data_samples])  # (N, C, H, W)

        # 3. 크기 보정 및 후처리
        pred_masks = F.interpolate(pred_masks, size=(2048, 2048), mode="bilinear", align_corners=False)  # 크기 보정
        pred_masks = torch.sigmoid(pred_masks)  # 확률값으로 변환
        pred_masks = (pred_masks > self.threshold).detach().cpu().numpy()  # Threshold 적용 후 NumPy 변환 (N, C, 2048, 2048)

        for pred_mask, image_name in zip(pred_masks, image_names):  # 배치 내 각 샘플에 대해 처리
            # Process each class mask
            for class_idx, mask in enumerate(pred_mask):
                rle = self._encode_mask_to_rle(mask)
                self.rles.append(rle)
                self.classes.append(f"{self.IND2CLASS[class_idx]}")
                self.filenames.append(f"{image_name}")

        self.results = []

    def compute_metrics(self, results) -> dict:
        
        self.inference(self.rles, self.filenames, self.classes)
        return {}
    
    def inference(self, rles, filenames, classes):

        df = pd.DataFrame({
                "image_name": filenames,
                "class": classes,
                "rle": rles,
            })
        df.to_csv("output.csv", index=False)
        print("Test results saved to output.csv")