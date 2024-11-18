from constants import CLASSES, IND2CLASS

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.classification import BinaryJaccardIndex
from utils import dice_coef, encode_mask_to_rle

import os
import pandas as pd

from model import load_model

class SegmentationModel(LightningModule):
    def __init__(self, model, criterion, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = learning_rate
        self.validation_dices = []
        self.best_dice = 0.0  # 최고 Dice 점수 초기화

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        logits = outputs.logits  # logits 추출
        loss = self.criterion(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        logits = outputs.logits  # logits 추출
        loss = self.criterion(logits, masks)
        
        # Dice coefficient 계산
        dice = self._dice_coefficient(logits, masks)
        self.validation_dices.append(dice.unsqueeze(0))  # 1차원 텐서로 추가
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice_batch", dice, prog_bar=True)
        return {"val_loss": loss, "dice": dice}

    def on_validation_epoch_end(self):
        if len(self.validation_dices) > 0:
            dices = torch.cat(self.validation_dices, dim=0)
            avg_dice = dices.mean()
            self.log("val_dice", avg_dice, prog_bar=True)
            
            # 최고 Dice 점수 갱신
            if avg_dice > self.best_dice:
                self.best_dice = avg_dice.item()
                self.log("best_dice", self.best_dice, prog_bar=True)
        else:
            self.log("val_dice", torch.tensor(0.0), prog_bar=True)
        
        self.validation_dices = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    @staticmethod
    def _dice_coefficient(outputs, targets, smooth=1e-5):
        outputs = torch.sigmoid(outputs)  # 예측값을 확률로 변환
        outputs = (outputs > 0.5).float()  # 이진화
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

    def on_validation_epoch_end(self):
        if len(self.validation_dices) > 0:
            dices = torch.cat(self.validation_dices, dim=0)
            avg_dice = dices.mean()

            # 클래스별 Dice 계산
            dices_per_class = dices.mean(dim=0)  # 각 클래스에 대한 평균 계산

            # 로그 출력 및 best_dice 업데이트
            self.log("val_dice", avg_dice, prog_bar=True)
            if avg_dice > self.best_dice:
                self.best_dice = avg_dice.item()
                self.log("best_dice", self.best_dice, prog_bar=True)

            # 클래스별 Dice 점수를 사전으로 저장
            dice_scores_dict = {f'val/{c}': d.item() for c, d in zip(CLASSES, dices_per_class)}
            self.log_dict(dice_scores_dict, prog_bar=False)
        else:
            self.log("val_dice", torch.tensor(0.0), prog_bar=True)

        self.validation_dices = []

    def test_step(self, batch, batch_idx):
        image_names, images = batch
        outputs = self(images)

        # 크기 보정
        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
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

    def configure_optimizers(self):  
        # Optimizer를 정의합니다.
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=1e-6)

        return optimizer
    