from model_lightning_palm import *

import numpy as np

import torch
import torch.nn.functional as F

import pandas as pd
from torchvision.transforms import ToTensor

from utils import encode_mask_to_rle, decode_rle_to_mask

from constants import PALM_CLASSES, IND2CLASS, CLASSES

class SegmentationModel_ensemble(SegmentationModel_palm):

    def __init__(self, gt_csv=None, model_weights=[0.5, 0.5], thresholds=[]):
        super().__init__()
        self.model_weights = torch.tensor(model_weights)
        self.thresholds = torch.tensor(thresholds)
        #self.model = self.load_model(architecture, encoder_name, encoder_weight)
        self.palm_crop_info = None
        if gt_csv is not None:
            self.palm_crop_info = self.get_palm_box(gt_csv)
        self.toTensor = ToTensor()
        
        self.palm_models = []
        self.models = []

    def set_model(self, palm_models, models):
        # 모델을 디바이스로 이동하고 평가 모드로 설정
        self.palm_models = [model.cuda() for model in palm_models]
        self.models = [model.cuda() for model in models]
    
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
        palm_outputs = torch.zeros((B, len(CLASSES), 1024, 1024)).to(images.device)

        for palm_model in self.palm_models:
            palm_output = palm_model(resized_images)
            palm_output = torch.sigmoid(palm_output)
            palm_outputs += palm_output
        palm_outputs /= len(self.palm_models)
        
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

    def general_batch(self, images):
        B = images.size(0)
        outputs = torch.zeros((B, len(CLASSES), 1024, 1024)).to(images.device)
        
        for model in self.models:
            general_output = model(images)
            general_output = torch.sigmoid(general_output)
            outputs += general_output

        outputs /= len(self.models)
        return outputs
    
    def test_step(self, batch, batch_idx):
        image_names, images = batch

        # Crop info 준비
        crop_infos = [self.palm_crop_info[image_name] for image_name in image_names]

        # 배치 단위 처리
        palm_outputs = self.ensemble_palm_batch(images, crop_infos)
        print("palm batch done")
        
        resized_images = F.interpolate(images, size=(1024, 1024), mode="bilinear", align_corners=False)
        general_outputs = self.general_batch(resized_images)
        general_outputs = F.interpolate(general_outputs, size=(2048, 2048), mode="bilinear", align_corners=False)
        
        print("general batch done")
        
        outputs = []
        for idx, classname in enumerate(CLASSES):
            if classname in PALM_CLASSES:
                class_output = palm_outputs[:, idx, :, :] * self.model_weights[0] + general_outputs[:, idx, :, :] * self.model_weights[1]
            else:
                class_output = general_outputs[:, idx, :, :]
            outputs.append((class_output > self.thresholds[idx]).detach().cpu().numpy())
        outputs = np.array(outputs)
        outputs = np.transpose(outputs, (1, 0, 2, 3))
        print(outputs.shape)
        
        # RLE 인코딩 및 파일명 생성
        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
                rle = encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")