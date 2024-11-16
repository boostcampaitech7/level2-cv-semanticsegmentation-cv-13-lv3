import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class CustomLightningModule(pl.LightningModule):
    def __init__(self, encoder_name, encoder_weights, classes, lr):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)
        preds = torch.sigmoid(outputs)
        dice = self.dice_coef(preds, masks)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        return [optimizer], [scheduler]

    @staticmethod
    def dice_coef(preds, targets, smooth=1.0):
        preds = preds > 0.5
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()