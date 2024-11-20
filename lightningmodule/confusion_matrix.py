# torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xraydataset import XRayDataset, split_data
from utils.utils import get_sorted_files_by_type, set_seed
from constants import TRAIN_DATA_DIR
from argparse import ArgumentParser
import albumentations as A
import os
import torch
from model_lightning import SegmentationModel
from lightning.pytorch import Trainer, seed_everything
import numpy as np
from omegaconf import OmegaConf
# ... existing code ...

def confusion_matrix(args):
    seed_everything(args.seed)
    set_seed(args.seed)
    
    # 체크포인트 로드
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트가 존재하지 않음 <{checkpoint_path}>")
    
    # 데이터 로더 설정
    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')
    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')
    _, valid_files = split_data(pngs, jsons)
    
    valid_dataset = XRayDataset(
        image_files=valid_files['filenames'],
        label_files=valid_files['labelnames'],
        transforms=A.Resize(args.input_size, args.input_size)
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=2,
        shuffle=False,
        num_workers=7,
        drop_last=False
    )

    # 모델 및 데이터 로더 설정
    criterion = nn.BCEWithLogitsLoss()
    seg_model = SegmentationModel(
        criterion=criterion,
        learning_rate=args.lr,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weight=args.encoder_weight,
        use_confusion_matrix=True
    )
    
    # Trainer 설정
    trainer = Trainer(
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if args.amp else 32,
    )

    # validation 실행
    trainer.validate(seg_model, dataloaders=valid_loader, ckpt_path=checkpoint_path)

# main 부분에 validation 옵션 추가
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    confusion_matrix(cfg)