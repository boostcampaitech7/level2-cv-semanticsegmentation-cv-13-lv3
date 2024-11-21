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
from omegaconf import OmegaConf

# 테스트를 수행하는 함수
def test_model(args):

    if args.resume_checkpoint_suffix == None:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt")
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}{args.resume_checkpoint_suffix}.ckpt")
    
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/base_config.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    test_model(cfg)