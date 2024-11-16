from configs.config_factory import get_config
from utils.Gsheet import Gsheet_param
from module.dataset import XRayDataModule
from module.trainer import CustomLightningModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch


def train_model():
    config = get_config()
    seed_everything(config.seed)

    data_module = XRayDataModule(
        image_root=config.image_root,
        label_root=config.label_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        valid_split=config.valid_split,
        transforms={"train": config.train_transforms, "valid": config.valid_transforms}
    )
    data_module.setup()

    seg_model = CustomLightningModule(
        model_name=config.model_name,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=29,
        lr=config.lr
    )

    trainer = Trainer(
        max_epochs=config.max_epoch,
        check_val_every_n_epoch=config.valid_interval,
        logger=WandbLogger(project=config.project_name, name=config.run_name),
        callbacks=[ModelCheckpoint(dirpath=config.checkpoint_dir, filename=f"{config.model_name}_best")],
        accelerator='gpu',
        devices=1
    )

    trainer.fit(seg_model, datamodule=data_module)


if __name__ == '__main__':
    train_model()