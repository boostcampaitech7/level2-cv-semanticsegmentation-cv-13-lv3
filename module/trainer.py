import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model_name, encoder_name, encoder_weights, classes, lr):
        super().__init__()
        self.save_hyperparameters()  # Hyperparameter 저장

        # SMP 모델 초기화
        smp_model_map = {
            "Unet++": "UnetPlusPlus",
        }
        if model_name in smp_model_map:
            self.model = getattr(smp, smp_model_map[model_name])(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=None,
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 손실 함수 정의
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}