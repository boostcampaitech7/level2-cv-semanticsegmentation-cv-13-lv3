import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model_name, encoder_name, encoder_weights, classes, lr):
        super(CustomLightningModule, self).__init__()
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.train_acc = MulticlassAccuracy(num_classes=classes)
        self.val_acc = MulticlassAccuracy(num_classes=classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        acc = self.train_acc(torch.sigmoid(outputs), masks.int())
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.val_acc(torch.sigmoid(outputs), masks.int())
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer