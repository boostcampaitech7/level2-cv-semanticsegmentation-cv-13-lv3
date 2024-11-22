from constants import CLASSES, IND2CLASS, PALM_CLASSES

import torch
import torch.optim as optim
import torch.nn.functional as F
from lightning import LightningModule
from utils import dice_coef, encode_mask_to_rle

import os
import pandas as pd
import numpy as np
from PIL import Image


from model import load_model

def crop_img_custom(image, min, max):

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

def pad_image(image):
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

class SegmentationModel(LightningModule):
    def __init__(self, criterion, learning_rate, thr=0.5, architecture="Unet", encoder_name="resnet50", encoder_weight="imagenet"):
        super(SegmentationModel, self).__init__()
        self.save_hyperparameters(ignore=['criterion'])  # criterion은 제외
        self.model = load_model(architecture, encoder_name, encoder_weight)
        self.model_palm = None
        self.criterion = criterion
        self.lr = learning_rate
        self.thr = thr
        self.best_dice = 0.0
        self.best_epoch = -1   # Best Epoch 초기화
        self.validation_dices = []  # validation_step 출력을 저장할 리스트

        self.rles = []
        self.filename_and_class = []

        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        _, images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        
        # 학습률 로깅
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False)
        
        # 손실 로깅
        self.log('train/loss', loss, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_idx):
        _, images, masks = batch
        outputs = self(images)
        
        # 크기 보정
        if outputs.size(-2) != masks.size(-2) or outputs.size(-1) != masks.size(-1):
            outputs = F.interpolate(outputs, size=masks.shape[-2:], mode="bilinear")

        loss = self.criterion(outputs, masks)
        self.log('val/loss', loss, prog_bar=True, on_step=True, on_epoch=False)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.thr)
        masks = masks
        
        dice = dice_coef(outputs, masks).detach().cpu()
        self.validation_dices.append(dice)  # dice score 저장
        return dice


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
            
        # WandB에 현재 Best Epoch 기록
        #self.log('best_epoch', self.best_epoch, logger=True)

        # Log Dice scores per class using WandB logger
        dice_scores_dict = {'val/' + c: d.item() for c, d in zip(CLASSES, dices_per_class)}
        self.log_dict(dice_scores_dict, on_epoch=True, logger=True)  # Log to WandB at the end of each epoch

        # 에폭이 끝나면 validation_dices 초기화
        self.validation_dices.clear()


    def _ensemble_palm(self, image_names, images, outputs):

        target_image_length = (1024, 1024)

        palm_output = []
        palm_images = []
        palm_metadata = dict(pos=[],scale=[])

        for output, image in zip(outputs, images):
            for c, segm in enumerate(output):
                if IND2CLASS[c] in PALM_CLASSES:
                    palm_output.append(segm)
            
                palm_images.append(image)
      
        y_indices, x_indices = np.where(palm_output == 1)
        # 최소 및 최대 좌표 계산
        min_x, max_x = x_indices.min(), x_indices.max()
        min_y, max_y = y_indices.min(), y_indices.max()

        modded_pos = (0,0)

        palm_images, pos = crop_img_custom(palm_images, min=(min_x, min_y), max=(max_x, max_y))
        modded_pos = -pos

        palm_images, pad_pos = pad_image(palm_images)
        modded_pos += pad_pos

        palm_metadata['pos'] = modded_pos
        palm_metadata['orig_size'] = palm_images.size
        palm_images = F.interpolate(palm_images, size=target_image_length, mode="bilinear")

        palm_outputs = self.model_palm(palm_images)
        palm_outputs = torch.sigmoid(palm_outputs)
        #palm_outputs = (palm_outputs > self.thr).detach().cpu().numpy()

        palm_outputs = F.interpolate(palm_outputs, size=palm_metadata['orig_size'], mode="bilinear")

        palm_outputs = pad_image_to_target(palm_outputs, palm_metadata['pos'])

        return palm_outputs


    def test_step(self, batch, batch_idx):
        image_names, images = batch
        outputs = self(images)

        # 크기 보정
        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.thr).detach().cpu().numpy()

        palm_outputs = self._ensemble_palm(image_names, images, outputs)

        ensemble_outputs = ensemble(outputs, palm_outputs)

        # RLE 인코딩 및 파일명 생성
        for output, image_name in zip(ensemble_outputs, image_names):
            for c, segm in enumerate(ensemble_outputs):
                rle = encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")


    def on_test_epoch_end(self):
        # 클래스 및 파일 이름 분리
        classes, filename = zip(*[x.split("_") for x in self.filename_and_class])
        image_name = [os.path.basename(f) for f in filename]

        # 결과를 DataFrame에 저장하고 CSV로 출력
        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": self.rles,
        })
        df.to_csv("output.csv", index=False)
        print("Test results saved to output.csv")
        
        
    def on_train_epoch_end(self):
        self.log('epoch', self.current_epoch)  # 에폭 번호를 로그로 기록


    def configure_optimizers(self):  
        # Optimizer 정의
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

        return optimizer
        
        # 옵티마이저와 스케줄러 반환
        #return [optimizer], [scheduler]
    