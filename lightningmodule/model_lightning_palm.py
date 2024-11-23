from model_lightning import SegmentationModel

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import pandas as pd
from torchvision.transforms import ToTensor

from utils import encode_mask_to_rle, decode_rle_to_mask

from constants import PALM_CLASSES, PALM_IND2CLASS

def preprocess_images_batch(images, min_pos, max_pos, inference_size=(1024, 1024)):
    """
    이미지 배치에서 크롭, 패딩, 리사이즈를 한 번에 수행합니다.
    Args:
        images: torch.Tensor (B, C, H, W)
        min_pos: torch.Tensor (B, 2) - 각 이미지의 크롭 최소 좌표 (x, y)
        max_pos: torch.Tensor (B, 2) - 각 이미지의 크롭 최대 좌표 (x, y)
        target_size: tuple - 리사이즈 대상 크기 (H, W)
    Returns:
        resized_images: torch.Tensor (B, C, target_H, target_W)
        crop_offsets: torch.Tensor (B, 2) - 크롭 좌표 오프셋 (x, y)
        pad_offsets: torch.Tensor (B, 2) - 패딩 좌표 오프셋 (x, y)
        original_sizes: torch.Tensor (B, 2) - 크롭된 이미지의 원본 크기 (H, W)
    """
    B, C, H, W = images.shape
    min_x, min_y = min_pos[:, 0], min_pos[:, 1]
    max_x, max_y = max_pos[:, 0], max_pos[:, 1]

    crop_w, crop_h = max_x - min_x, max_y - min_y
    

    cropped_images = []
    crop_offsets = []
    original_sizes = []

    # 1. Crop Images
    for i in range(B):
        # 이미지 크기에 따라 여백 비율 조정
        width_margin = int((H - crop_w[i]) * 0.005)  
        height_margin = int((W - crop_h[i]) * 0.005)  

        cropped = images[i, :, 
                         min_y[i]-height_margin:max_y[i]+height_margin, 
                         min_x[i]-width_margin:max_x[i]+width_margin]  # 크롭
        cropped_images.append(cropped)
        original_sizes.append((cropped.shape[1], cropped.shape[2]))  # (H, W)
        print(f'cropped : {cropped.shape}')
        crop_offsets.append((min_x[i]-width_margin, min_y[i]-height_margin))
        print(f'cropped offset : {(min_x[i]-width_margin, min_y[i]-height_margin)}')

    # 2. Pad to Square
    max_crop_H = max([img.shape[1] for img in cropped_images])  # 최대 높이
    max_crop_W = max([img.shape[2] for img in cropped_images])  # 최대 너비
    max_dim = max(max_crop_H, max_crop_W)  # 정사각형 기준 크기

    padded_images = []
    pad_offsets = []

    for cropped in cropped_images:
        pad_H = max_dim - cropped.shape[1]
        pad_W = max_dim - cropped.shape[2]
        # Ensure symmetric padding
        padded = F.pad(
            cropped,
            (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2),  # (left, right, top, bottom)
            mode="constant",
            value=0,
        )
        padded_images.append(padded)
        pad_offsets.append((pad_W // 2, pad_H // 2))
        print(f'pad offset : {(pad_W // 2, pad_H // 2)}')

    # 3. Stack Padded Images
    padded_images = torch.stack(padded_images, dim=0)  # (B, C, max_dim, max_dim)

    # 4. Resize to Target Size
    resized_images = F.interpolate(padded_images, size=inference_size, mode="bilinear", align_corners=False)

    return resized_images, torch.tensor(crop_offsets), torch.tensor(pad_offsets), torch.tensor(original_sizes)

def restore_to_original_sizes(predictions, original_sizes, crop_offsets, pad_offsets, target_size=(2048, 2048)):
    """
    모델의 예측을 원본 크기로 복원합니다.
    Args:
        predictions: torch.Tensor (B, C, target_H, target_W)
        original_sizes: torch.Tensor (B, 2)
        crop_offsets: torch.Tensor (B, 2)
        pad_offsets: torch.Tensor (B, 2)
        target_size: tuple
    Returns:
        restored_outputs: torch.Tensor (B, C, target_H, target_W)
    """
    B, C, _, _ = predictions.shape
    target_H, target_W = target_size

    restored_outputs = torch.zeros((B, C, target_H, target_W), device=predictions.device)
    for i in range(B):
        # Resize back to cropped size
        orig_H, orig_W = original_sizes[i]
        resized_output = F.interpolate(
            predictions[i].unsqueeze(0), size=(orig_H, orig_W), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Pad back to original position
        pad_x, pad_y = pad_offsets[i]
        crop_x, crop_y = crop_offsets[i]
        restored_outputs[i, :, crop_y + pad_y : crop_y + pad_y + orig_H, crop_x + pad_x : crop_x + pad_x + orig_W] = resized_output

    return restored_outputs

def crop_img_by_minmax(image, min_pos, max_pos):

    # 최소 및 최대 x, y 좌표 계산
    min_x, min_y = min_pos
    max_x, max_y = max_pos
    
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

    #print(new_min_x, new_max_x, new_min_y, new_max_y)

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
    
    new_pos = (x_offset, y_offset)

    # 원본 이미지를 가운데에 붙여넣기
    padded_image.paste(image, new_pos)
    
    return padded_image, new_pos

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
        self.toTensor = ToTensor()
    
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

            crop_info[image_name]['min'] = (min_x//2, min_y//2)
            crop_info[image_name]['max'] = (max_x//2, max_y//2)
            
        return crop_info

    def ensemble_palm_batch(self, images, crop_infos):
        """
        배치 단위로 이미지를 처리합니다.
        Args:
            images: torch.Tensor (B, C, H, W)
            crop_infos: List of dicts containing 'min' and 'max' for each image
        Returns:
            palm_outputs: torch.Tensor (B, C, H, W)
        """
        B = images.size(0)
        inference_size = (1024, 1024)

        # Crop 정보 준비
        min_pos = torch.tensor([info['min'] for info in crop_infos], device=images.device)
        max_pos = torch.tensor([info['max'] for info in crop_infos], device=images.device)

        # 1. Preprocess Images (Crop, Pad, Resize)
        resized_images, crop_offsets, pad_offsets, original_sizes = preprocess_images_batch(
            images, min_pos, max_pos, inference_size=inference_size
        )
        # 4. Forward pass
        palm_outputs = self(resized_images)

        # 3. Restore to Original Sizes
        restored_outputs = restore_to_original_sizes(palm_outputs, original_sizes, crop_offsets, pad_offsets)

        return restored_outputs

    def test_step(self, batch, batch_idx):
        image_names, images = batch

        # Crop info 준비
        crop_infos = [self.palm_crop_info[image_name] for image_name in image_names]

        # 배치 단위 처리
        outputs = self.ensemble_palm_batch(images, crop_infos)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.thr).detach().cpu().numpy()

        # RLE 인코딩 및 파일명 생성
        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(f"{PALM_IND2CLASS[c]}_{image_name}")