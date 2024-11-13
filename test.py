# torch
from torch.utils.data import DataLoader


from xraydataset import XRayDataset
from utils import get_sorted_files_by_type

from constants import TEST_DATA_DIR

from argparse import ArgumentParser

import albumentations as A

import os
import torch

from model_lightning import SegmentationModel

from lightning.pytorch import Trainer

import numpy as np

# 테스트를 수행하는 함수
def test_model(args):

    # 모델 및 체크포인트 경로 설정
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    
    seg_model = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path, criterion=None, learning_rate=None)

    # 데이터 로드 및 테스트
    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    test_dataset = XRayDataset(image_files=np.array(pngs), transforms=A.Resize(args.input_size, args.input_size))  # 원하는 입력 크기로 조정
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    # Trainer 설정
    trainer = Trainer(accelerator='gpu', devices=1 if torch.cuda.is_available() else None)
    
    # 테스트 실행
    trainer.test(seg_model, dataloaders=test_loader)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str,default="./checkpoints")
    parser.add_argument('--checkpoint_file', type=str,default="fcn_resnet50_best_model.ckpt")
    
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument("--input_size", type=int, default=512)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    test_model(args)