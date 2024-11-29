from model_lightning import SegmentationModel

import numpy as np

import torch
import torch.nn.functional as F

import pandas as pd
from torchvision.transforms import ToTensor

from utils import encode_mask_to_rle, decode_rle_to_mask, label2rgb

from constants import PALM_CLASSES, IND2CLASS

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

        cropped_x = min_x[i] - width_margin
        cropped_y = min_y[i] - height_margin

        cropped = images[i, :, 
                         cropped_y:max_y[i]+height_margin, 
                         cropped_x:max_x[i]+width_margin]  # 크롭
        cropped_images.append(cropped)
        crop_offsets.append((cropped_x, cropped_y))
        print(f'cropped offset : {(cropped_x, cropped_y)}')

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
        original_sizes.append((padded.shape[1], padded.shape[2]))  # (H, W)
        pad_offsets.append((pad_W // 2, pad_H // 2))

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

        # Determine the position in the target canvas
        crop_x, crop_y = crop_offsets[i]
        pad_x, pad_y = pad_offsets[i]

        # Adjust the position: take crop_offsets into account, subtract pad_offsets
        start_x = crop_x - pad_x
        start_y = crop_y - pad_y
        end_x = start_x + orig_W
        end_y = start_y + orig_H

        # Ensure the coordinates are valid within the canvas size
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(target_W, end_x)
        end_y = min(target_H, end_y)


        print(f'restored offset : {(start_x, end_x)}')

        # Place the resized output into the restored canvas
        restored_outputs[i, :, start_y:end_y, start_x:end_x] = resized_output[:, :end_y-start_y, :end_x-start_x]

    return restored_outputs

class SegmentationModel_palm(SegmentationModel):

    def __init__(self, gt_csv=None, architecture="UperNet", encoder_name="efficientnet-b7", encoder_weight="imagenet"):
        super().__init__(architecture=architecture, encoder_name=encoder_name, encoder_weight=encoder_weight)
        self.palm_crop_info = None
        if gt_csv is not None:
            self.palm_crop_info = self.get_palm_box(gt_csv)
        self.toTensor = ToTensor()

    def set_model(self,model):
        self.model = model
    
    def get_palm_box(self, csv_path):
        df = pd.read_csv(csv_path)

        crop_info = dict()

        # 그룹화하여 처리
        grouped = df.groupby('image_name')
        for idx, (image_name, group) in enumerate(grouped):
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

        # 1,2,3. Preprocess Images (Crop, Pad, Resize)
        resized_images, crop_offsets, pad_offsets, original_sizes = preprocess_images_batch(
            images, min_pos, max_pos, inference_size=inference_size
        )

        print(f'interpolated size (should be 1k) : {(resized_images.shape)}')

        # 4. Forward pass
        palm_outputs = self(resized_images)

        # 3. Restore to Original Sizes
        restored_outputs = restore_to_original_sizes(palm_outputs, original_sizes, crop_offsets, pad_offsets)

        return restored_outputs

    def on_validation_epoch_end(self):
            dices = torch.cat(self.validation_dices, 0)
            dices_per_class = torch.mean(dices, 0)
            avg_dice = torch.mean(dices_per_class).item()
            
            # 로그와 체크포인트 저장을 위한 모니터링 지표로 사용
            self.log('val/dice', avg_dice, prog_bar=True)

            # Best Dice 및 Best Epoch 갱신
            if avg_dice > self.best_dice:
                self.best_dice = avg_dice
                self.best_epoch = self.current_epoch  # Best Epoch 갱신
                print(f"Best performance improved: {self.best_dice:.4f} at Epoch: {self.best_epoch}")
                
            # Log Dice scores per class using WandB logger
            dice_scores_dict = {'val/' + c: d.item() for c, d in zip(PALM_CLASSES, dices_per_class)}
            self.log_dict(dice_scores_dict, on_epoch=True, logger=True)  # Log to WandB at the end of each epoch

            # 에폭이 끝나면 validation_dices 초기화
            self.validation_dices.clear()

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
                self.filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")