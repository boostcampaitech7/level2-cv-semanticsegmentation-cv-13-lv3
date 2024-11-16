import os
import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from module.model import build_model

class CustomLightningModule(pl.LightningModule):
    def __init__(self, stage, num_classes=2, lr=0.001):
        super().__init__()
        self.stage = stage
        self.model = build_model(stage, num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc(preds, labels))
        self.log("train_f1", self.train_f1(preds, labels))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.model(images)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(preds, labels))
        self.log("val_f1", self.val_f1(preds, labels))
        return loss