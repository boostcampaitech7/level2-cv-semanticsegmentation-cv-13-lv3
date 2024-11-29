from constants import CLASSES, IND2CLASS, PALM_CLASSES

import torch
import torch.optim as optim
import torch.nn.functional as F
from lightning import LightningModule
from utils import dice_coef, encode_mask_to_rle

import os
import pandas as pd

from model import load_model

class SegmentationModel(LightningModule):
    def __init__(self, criterion = None, learning_rate = None, thr=0.5, architecture="UperNet", encoder_name="efficientnet-b7", encoder_weight="imagenet"):
        super(SegmentationModel, self).__init__()
        self.model = load_model(architecture, encoder_name, encoder_weight)
        self.criterion = criterion
        self.lr = learning_rate
        self.thr = thr
        self.best_dice = 0.0
        self.best_epoch = -1   # Best Epoch 초기화
        self.validation_dices = []  # validation_step 출력을 저장할 리스트

        self.rles = []
        self.filename_and_class = []

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

        # Log Dice scores per class using WandB logger
        dice_scores_dict = {'val/' + c: d.item() for c, d in zip(PALM_CLASSES, dices_per_class)}
        self.log_dict(dice_scores_dict, on_epoch=True, logger=True)  # Log to WandB at the end of each epoch

        # 에폭이 끝나면 validation_dices 초기화
        self.validation_dices.clear()

    def test_step(self, batch, batch_idx):
        image_names, images = batch
        outputs = self(images)

        # 크기 보정
        outputs = F.interpolate(outputs, size=(1024, 1024), mode="bilinear")
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.thr).detach().cpu().numpy()

        # RLE 인코딩 및 파일명 생성
        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
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
        optimizer = optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        
        # 옵티마이저와 스케줄러 반환
        return [optimizer], [scheduler]
    
