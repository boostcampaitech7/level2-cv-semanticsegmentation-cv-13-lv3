import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from lightning.pytorch import LightningModule

class CustomLightningModule(LightningModule):
    def __init__(self, model, num_classes, lr):
        super(CustomLightningModule, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)