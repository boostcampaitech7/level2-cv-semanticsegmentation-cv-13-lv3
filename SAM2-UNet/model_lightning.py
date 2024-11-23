from constants import CLASSES, IND2CLASS

import torch
import torch.nn.functional as F
from torch import nn
from lightning import LightningModule
from SAM2UNet import SAM2UNet
from utils import dice_coef, encode_mask_to_rle  # 기존 dice_coef 활용

import os
import pandas as pd

class SegmentationModel(LightningModule):
    def __init__(self, criterion, num_classes=29, learning_rate=1e-3, checkpoint_path=None):
        super(SegmentationModel, self).__init__()
        self.model = SAM2UNet(num_classes=num_classes, checkpoint_path=checkpoint_path)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()  # 기본 손실 함수
        self.lr = learning_rate
        self.loss_weights = [1.0, 0.5, 0.25]  # 메인 출력과 보조 출력에 대한 가중치 설정
        self.thr = 0.5
        self.best_dice = 0.0
        self.best_epoch = -1
        self.validation_dices = [] # validation_step에서 Dice Score 저장

        self.rles = []
        self.filename_and_class = []


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        _, images, masks = batch
        outputs, aux1, aux2 = self.model(images)

        # 멀티스케일 손실 계산
        main_loss = self.criterion(outputs, masks)
        aux_loss1 = self.criterion(aux1, masks)
        aux_loss2 = self.criterion(aux2, masks)
        total_loss = (
            self.loss_weights[0] * main_loss +
            self.loss_weights[1] * aux_loss1 +
            self.loss_weights[2] * aux_loss2
        )

        # 학습률 로깅
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False)

        # 손실 로깅
        self.log("train/loss", total_loss, prog_bar=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        _, images, masks = batch
        outputs, aux1, aux2 = self.model(images)

        # 멀티스케일 손실 계산
        main_loss = self.criterion(outputs, masks)
        aux_loss1 = self.criterion(aux1, masks)
        aux_loss2 = self.criterion(aux2, masks)
        total_loss = (
            self.loss_weights[0] * main_loss +
            self.loss_weights[1] * aux_loss1 +
            self.loss_weights[2] * aux_loss2
        )

        # 손실 로깅
        self.log("val/loss", total_loss, prog_bar=True)

        # Dice 계산
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.thr)
        dice = dice_coef(outputs, masks).detach().cpu()
        self.validation_dices.append(dice)

        return dice


    def on_validation_epoch_end(self):
        # 모든 validation_step에서 계산된 Dice Score를 평균
        dices = torch.cat(self.validation_dices, 0)
        dices_per_class = torch.mean(dices, 0)
        avg_dice = torch.mean(dices_per_class).item()

        # Best Dice Score 업데이트
        if avg_dice > self.best_dice:
            self.best_dice = avg_dice
            self.best_epoch = self.current_epoch
            print(f"\nNew Best Dice: {self.best_dice:.4f} at Epoch {self.best_epoch}\n")

        # avg_dice를 val/dice로 로깅
        self.log("val/dice", avg_dice, prog_bar=True) 

        # 각 클래스별 Dice도 로깅
        dice_scores_dict = {'val/' + c: d.item() for c, d in zip(CLASSES, dices_per_class)}
        self.log_dict(dice_scores_dict, on_epoch=True, logger=True)

        # Epoch이 끝날 때 validation_dices 초기화
        self.validation_dices.clear()
        
        
    def test_step(self, batch, batch_idx):
        image_names, images = batch
        outputs, _, _ = self.model(images)

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
        classes, filenames = zip(*[x.split("_") for x in self.filename_and_class])
        image_names = [os.path.basename(f) for f in filenames]

        # 결과를 DataFrame에 저장하고 CSV로 출력
        df = pd.DataFrame({
            "image_name": image_names,
            "class": classes,
            "rle": self.rles,
        })
        output_path = "output_test.csv"
        df.to_csv(output_path, index=False)
        print(f"Test results saved to {output_path}")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer
