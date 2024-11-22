from model_lightning import SegmentationModel

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import pandas as pd

from utils import encode_mask_to_rle, decode_rle_to_mask

from constants import PALM_CLASSES, PALM_IND2CLASS

def crop_img_by_minmax(image, min, max):

    # 최소 및 최대 x, y 좌표 계산
    min_x, min_y = min
    max_x, max_y = max
    
    h, w = image.size[1], image.size[0]  
    crop_w, crop_h = max_x - min_x, max_y - min_y
    
    # 이미지 크기에 따라 여백 비율 조정
    width_margin = int((w - crop_w) * 0.005)  
    height_margin = int((h - crop_h) * 0.005)  

    # 여백을 포함하여 새로운 크롭 좌표 설정
    new_min_x = max(0, min_x - width_margin)
    new_max_x = min(w, max_x + width_margin)
    new_min_y = max(0, min_y - height_margin)
    new_max_y = min(h, max_y + height_margin)

    print(new_min_x, new_max_x, new_min_y, new_max_y)

    # 이미지 크롭
    image = image.crop([new_min_x, new_min_y, new_max_x, new_max_y])

    return image, (min_x, min_y)

def pad_to_sqaure(image):
    """
    이미지의 긴 쪽을 기준으로 짧은 쪽을 패딩하여 정사각형으로 만드는 함수
    이미지는 가운데 정렬되고 패딩은 양쪽에 균등하게 적용됨
    
    Args:
        image (PIL.Image): 입력 이미지
        
    Returns:
        PIL.Image: 패딩된 정사각형 이미지
    """
    # 이미지 크기 가져오기
    width, height = image.size
    
    # 긴 쪽 길이 찾기
    max_size = max(width, height)
    
    # 새로운 이미지 생성 (검은색 배경)
    padded_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))
    
    # 이미지를 가운데 위치시키기 위한 오프셋 계산
    x_offset = (max_size - width) // 2
    y_offset = (max_size - height) // 2
    
    # 원본 이미지를 가운데에 붙여넣기
    padded_image.paste(image, (x_offset, y_offset))
    
    return padded_image

# x, y 만큼의 패딩을 추가하고, 최종적으로 2048x2048로 만들기
def pad_image_to_target(image, pad, target_size=(2048, 2048)):

    pad_x, pad_y = pad

    # 초기 패딩 추가
    padded_image = np.pad(
        image, 
        ((pad_y, 0), (pad_x, 0)),  # 위쪽 pad_y, 왼쪽 pad_x 추가
        mode='constant',
        constant_values=0  # 패딩값은 0
    )
    
    # 목표 크기에 맞춰 오른쪽과 아래쪽에 추가 패딩
    height, width = padded_image.shape
    target_height, target_width = target_size
    
    if height > target_height or width > target_width:
        raise ValueError("이미지가 목표 크기를 초과합니다.")
    
    bottom_padding = target_height - height
    right_padding = target_width - width
    
    final_padded_image = np.pad(
        padded_image, 
        ((0, bottom_padding), (0, right_padding)),  # 아래쪽과 오른쪽 패딩 추가
        mode='constant',
        constant_values=0
    )
    
    return final_padded_image


class SegmentationModel_palm(SegmentationModel):

    def __init__(self, gt_csv, architecture="UperNet", encoder_name="efficientnet-b7", encoder_weight="imagenet"):
        super().__init__(architecture=architecture, encoder_name=encoder_name, encoder_weight=encoder_weight)
        self.palm_crop_info = self.get_palm_box(gt_csv)
    
    def ensemble_palm(self, image, crop_info):

        target_image_length = (1024, 1024)

        palm_metadata = dict(pos=[],scale=[])

        modded_pos = (0,0)

        print(image.shape)
        image = image.squeeze(0)

        image = Image.fromarray(image)

        image, pos = crop_img_by_minmax(image, min=crop_info['min'], max=crop_info['max'])
        modded_pos = -pos

        image, pad_pos = pad_to_sqaure(image)
        modded_pos += pad_pos

        palm_metadata['pos'] = modded_pos
        palm_metadata['orig_size'] = image.size
        image = F.interpolate(image, size=target_image_length, mode="bilinear")

        palm_output = self(image)

        palm_output = F.interpolate(palm_output, size=palm_metadata['orig_size'], mode="bilinear")
        palm_output = torch.sigmoid(palm_output)
        palm_outputs = (palm_outputs > self.thr)

        palm_output = pad_image_to_target(palm_output, palm_metadata['pos'])

        return palm_output

    def get_palm_box(self, csv_path):
        df = pd.read_csv(csv_path)

        crop_info = dict()

        # 그룹화하여 처리
        grouped = df.groupby('image_name')
        for image_name, group in grouped:
            crop_info[image_name] = dict()
            masks = []
            for _, row in group.iterrows():
                classname = row['class']
                if classname in PALM_CLASSES:
                    rle = row['rle']
                    if isinstance(rle, str):
                        mask = decode_rle_to_mask(rle, 2048, 2048)
                        masks.append(mask)

            combined_mask = np.logical_or.reduce(masks)

            y_indices, x_indices = np.where(combined_mask == True)
            # 최소 및 최대 좌표 계산
            min_x, max_x = x_indices.min(), x_indices.max()
            min_y, max_y = y_indices.min(), y_indices.max()

            crop_info[image_name]['min'] = (min_x, min_y)
            crop_info[image_name]['max'] = (max_x, max_y)
            
        return crop_info

    def test_step(self, batch, batch_idx):
        image_names, images = batch

        mask_outputs = []

        for image_name, image in zip(image_names, images): 
            mask_outputs.append(self.ensemble_palm(image.detach().cpu().numpy(), self.palm_crop_info[image_name]))

        # 크기 보정
        outputs = F.interpolate(outputs, size=(1024, 1024), mode="bilinear")
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.thr)

        # RLE 인코딩 및 파일명 생성
        for mask_output, image_name in zip(mask_outputs, image_names):
            for c, segm in enumerate(mask_output):
                rle = encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(f"{PALM_IND2CLASS[c]}_{image_name}")