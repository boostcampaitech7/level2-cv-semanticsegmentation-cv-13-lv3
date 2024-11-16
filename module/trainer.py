import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, lr):
        super(CustomLightningModule, self).__init__()
        self.model = model
        self.lr = lr
        self.train_acc = MulticlassAccuracy(num_classes=self.model.classes, average="macro")
        self.valid_acc = MulticlassAccuracy(num_classes=self.model.classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, masks)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        acc = self.train_acc(outputs.argmax(dim=1), masks)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, masks)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        acc = self.valid_acc(outputs.argmax(dim=1), masks)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer