# torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xraydataset import XRayDataset, split_data
from utils.utils import get_sorted_files_by_type, set_seed
from constants import TRAIN_DATA_DIR
from argparse import ArgumentParser, Namespace
import albumentations as A
import os
import torch
from model_lightning import SegmentationModel
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
from utils.Gsheet import Gsheet_param

# 모델 학습과 검증을 수행하는 함수
def train_model(args):
    args_dict = OmegaConf.to_container(args, resolve=True)
    run_name = args_dict.pop('run_name', None)
    project_name = args_dict.pop('project_name', None)
    seed_everything(args.seed)
    set_seed(args.seed)
    # wandb.init(project=args.project_name, name=args.run_name, config=args_dict)

    wandb_logger = WandbLogger(project=project_name, name=run_name, config=args_dict)

    # model = models.segmentation.fcn_resnet50(pretrained=True)
    # # output class 개수를 dataset에 맞도록 수정합니다.
    # model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.checkpoint_file,
        monitor='val/loss',  # 검증 손실을 기준으로 체크포인트 저장
        mode='min',
        save_top_k=3,
        save_weights_only=True  # WandB에 업로드할 용도로 모델 가중치만 저장
    )
    
    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    train_files, valid_files = split_data(pngs, jsons)

    train_dataset = XRayDataset(image_files=train_files['filenames'], label_files=train_files['labelnames'], transforms=A.Resize(args.input_size, args.input_size))
    valid_dataset = XRayDataset(image_files=valid_files['filenames'], label_files=valid_files['labelnames'], transforms=A.Resize(args.input_size, args.input_size))

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=7,
        drop_last=False
    )

    # Loss function을 정의합니다.
    criterion = nn.BCEWithLogitsLoss()

    # 체크포인트 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.checkpoint_file,
        monitor='val/dice',
        mode='max',
        save_top_k=3
    )

    # 모델 초기화
    seg_model = SegmentationModel(criterion=criterion, learning_rate=args.lr)

    # Trainer 설정
    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epoch,
        check_val_every_n_epoch=args.valid_interval,
        callbacks=[checkpoint_callback],
        accelerator='gpu', 
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if args.amp else 32
    )

    # 학습 시작
    trainer.fit(seg_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    wandb.save(os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/base_config.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    train_model(cfg)
    Gsheet_param(cfg)
