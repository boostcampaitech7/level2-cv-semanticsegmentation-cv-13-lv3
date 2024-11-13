# torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xraydataset import XRayDataset, split_data
from utils import get_sorted_files_by_type

from constants import TRAIN_DATA_DIR, WANDB_PROJECT_NAME

from argparse import ArgumentParser

import albumentations as A

import os
import torch

from model_lightning import SegmentationModel

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# 모델 학습과 검증을 수행하는 함수
def train_model(args):

    seed_everything(args.seed)

    config = args.__dict__
    run_name = config.pop('run_name', None)  # 'run_name'이 있으면 가져오고 없으면 None

    wandb_logger = WandbLogger(project=WANDB_PROJECT_NAME, name=run_name, config=config)

    # model = models.segmentation.fcn_resnet50(pretrained=True)
    # # output class 개수를 dataset에 맞도록 수정합니다.
    # model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)

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
        filename='fcn_resnet50_best_model',
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
        devices=1 if torch.cuda.is_available() else None
    )

    # 학습 시작
    trainer.fit(seg_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str,default="./checkpoints")
    parser.add_argument('--checkpoint_file', type=str,default="fcn_resnet50_best_model.pt")
    
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--run_name', type=str)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument("--input_size", type=int, default=512)

    # parser.add_argument("--amp", action="store_true", help="mixed precision")
 
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--valid_interval', type=int, default=1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    train_model(args)