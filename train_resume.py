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
from lightning.pytorch.utilities import rank_zero_info
from utils.Gsheet import Gsheet_param

def train_model(args):
    args_dict = OmegaConf.to_container(args, resolve=True)
    run_name = args_dict.pop('run_name', None)
    project_name = args_dict.pop('project_name', None)
    seed_everything(args.seed)

    # 체크포인트 경로 설정
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt")

    if args.resume:
        # Resume 설정 확인
        if os.path.exists(checkpoint_path):
            print(f"Resume : <{checkpoint_path}> 체크포인트에서 학습 재개")
        else:
            raise FileNotFoundError(f"Resume : 체크포인트가 존재하지 않음 <{checkpoint_path}>")
    else:
        checkpoint_path = None
        print("No Resume : 새로운 학습 시작")

    # WandB 설정
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        config=args_dict,
        resume="must" if args.resume else None,  # Resume=True일 때만 이어서 기록
        id=args.wandb_id if args.resume else None  # Resume=True일 때만 WandB ID 설정
    )

    # 데이터 로드
    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')
    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')
    train_files, valid_files = split_data(pngs, jsons)

    train_dataset = XRayDataset(
        image_files=train_files['filenames'],
        label_files=train_files['labelnames'],
        transforms=A.Resize(args.input_size, args.input_size)
    )
    valid_dataset = XRayDataset(
        image_files=valid_files['filenames'],
        label_files=valid_files['labelnames'],
        transforms=A.Resize(args.input_size, args.input_size)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=7,
        drop_last=False
    )

    # 모델 및 Trainer 설정
    criterion = nn.BCEWithLogitsLoss()
    seg_model = SegmentationModel(
        criterion=criterion,
        learning_rate=args.lr,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weight=args.encoder_weight
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.checkpoint_file,
        monitor='val/dice',
        mode='max',
        save_top_k=3
    )

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epoch,
        check_val_every_n_epoch=args.valid_interval,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if args.amp else 32,
        #resume_from_checkpoint=checkpoint_path  # 체크포인트에서 학습 재개
    )

    # 학습 시작
    trainer.fit(seg_model, 
                train_dataloaders=train_loader, 
                val_dataloaders=valid_loader,
                ckpt_path=checkpoint_path if args.resume else None  # 체크포인트 경로 전달
                )


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
