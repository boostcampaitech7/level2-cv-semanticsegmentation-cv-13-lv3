import os
import numpy as np

from torch import Tensor
from typing import Any, Sequence
from mmseg.registry import METRICS
from mmengine.evaluator import BaseMetric 
from mmengine.structures import BaseDataElement

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
                 collect_device='cpu',
                 prefix=None,
                 **kwargs):
        self.rles = []
        self.filename_and_class = []
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

        for data_sample in data_samples:
            # Retrieve metadata
            image_path = data_sample['img_path']
            image_name = os.path.basename(image_path)

            # Retrieve prediction
            pred_mask = data_sample['pred_sem_seg']['data'].cpu().numpy()  # (C, H, W)
            pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Apply thresholding
            
            # Process each class mask
            for class_idx, mask in enumerate(pred_mask):
                    rle = self._encode_mask_to_rle(mask)
                    self.rles.append(rle)
                    self.filename_and_class.append(f"{self.IND2CLASS[class_idx]}_{image_name}")

        self.results = []

    def compute_metrics(self, results) -> dict:
        return {}
    
    def get_results(self):
        return self.rles, self.filename_and_class