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
from constants import CLASSES, CLASS2IND, TRAIN_DATA_DIR, TEST_DATA_DIR
import os


def train_model():
    # Config 설정 및 시드 초기화
    config = get_config("UNet")  # 모델 이름에 맞게 설정을 로드
    seed_everything(config.seed)

    # 이미지 및 라벨 파일 로드
    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(config.image_root)
        for f in files if f.endswith(".png")
    ]
    label_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(config.label_root)
        for f in files if f.endswith(".json")
    ]

    if not image_files or not label_files:
        raise ValueError(
            f"No image files or label files found in the specified directories. "
            f"Check if {config.image_root} and {config.label_root} contain .png and .json files respectively."
        )

    # Train / Validation 데이터 분리
    train_data, valid_data = split_data(image_files, label_files, K=5, valid_idx=1)

    # Transform 설정
    train_transform = config.get_transforms(mode="train")
    valid_transform = config.get_transforms(mode="valid")

    # Dataset 생성
    train_dataset = XRayDataset(
        image_files=train_data["filenames"],
        label_files=train_data["labelnames"],
        transforms=train_transform
    )
    valid_dataset = XRayDataset(
        image_files=valid_data["filenames"],
        label_files=valid_data["labelnames"],
        transforms=valid_transform
    )

    # DataLoader 생성
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

    # 모델 초기화
    seg_model = CustomLightningModule(
        model=build_model(
            model_name=config.model_name,
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            num_classes=len(CLASSES),
        ),
        num_classes=len(CLASSES),
        lr=config.lr,
    )

    # WandB 로깅 설정
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=config.run_name,
        config=vars(config),
    )

    # Trainer 초기화
    trainer = Trainer(
        max_epochs=config.max_epoch,
        check_val_every_n_epoch=config.valid_interval,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=config.checkpoint_dir, 
                filename=f"{config.model_name}_best"
            )
        ],
        accelerator="gpu",
        devices=1,
    )

    # 학습 실행
    trainer.fit(seg_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # GSheet 파라미터 저장
    Gsheet_param(config)

if __name__ == "__main__":
    train_model()