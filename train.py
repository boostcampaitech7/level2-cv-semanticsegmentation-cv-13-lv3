import os
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from configs.config_factory import build_model,get_config
from configs.base_config import BaseConfig
from module.trainer import CustomLightningModule
from xraydataset import XRayDataset, split_data
from torch.utils.data import DataLoader
from utils.Gsheet import Gsheet_param

def train_model():
    config = get_config("Unet++")
    seed_everything(config.seed)

    image_files = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(config.image_root)  
        for f in files if f.endswith('.png')
    ]
    label_files = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(config.label_root)  
        for f in files if f.endswith('.json')
    ]

    if len(image_files) == 0 or len(label_files) == 0:
        raise ValueError(f"No image files or label files found in the specified directories. "
                         f"Check if {config.image_root} and {config.label_root} contain .png and .json files respectively.")

    train_data, valid_data = split_data(image_files, label_files, K=5, valid_idx=1)

    train_transform = config.get_transforms(mode="train")
    valid_transform = config.get_transforms(mode="valid")

    train_dataset = XRayDataset(
        train_data['filenames'], train_data['labelnames'], transforms=train_transform
    )
    valid_dataset = XRayDataset(
        valid_data['filenames'], valid_data['labelnames'], transforms=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    seg_model = CustomLightningModule(
        model=build_model(
            model_name=config.model_name,
            encoder_name="resnet34",
            encoder_weights="imagenet",
            num_classes=29,
        ),
        lr=config.lr,
    )

    wandb_logger = WandbLogger(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
    )

    trainer = Trainer(
        max_epochs=config.max_epoch,
        check_val_every_n_epoch=config.valid_interval,
        logger=wandb_logger,
        callbacks=[ModelCheckpoint(dirpath=config.checkpoint_dir, filename=f"{config.model_name}_best")],
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(seg_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    Gsheet_param(config)

if __name__ == "__main__":
    train_model()