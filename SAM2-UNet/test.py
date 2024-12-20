# torch
from torch.utils.data import DataLoader
from xraydataset import XRayDataset, split_data
from utils import get_sorted_files_by_type
from constants import TEST_DATA_DIR, TRAIN_DATA_DIR
from argparse import ArgumentParser
import albumentations as A
import os
import torch
from model_lightning import SegmentationModel
#from model_lightning import SegmentationModel_palm
from lightning.pytorch import Trainer
import numpy as np
from omegaconf import OmegaConf

# 테스트를 수행하는 함수
def test_model(args):

    if args.ckpt:
        # 모델 및 체크포인트 경로 설정
        if args.resume_checkpoint_suffix == None:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt")
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}{args.resume_checkpoint_suffix}.ckpt")
        seg_model = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path, criterion=None, learning_rate=None)
    else:
        pt_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}-final.pt")
        seg_model = torch.load(pt_path)  # 전체 모델 저장된 경우

    image_files = None
    # 데이터 로드 및 테스트
    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    pngs = get_sorted_files_by_type(image_root, 'png')

    image_files = np.array(pngs)

    test_dataset = XRayDataset(image_files=image_files,
                               transforms=A.Compose([        
                                                    A.Resize(args.input_size, args.input_size),         
                                                    A.Normalize(normalization='min_max', p=1.0)]))  # 원하는 입력 크기로 조정
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
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--ckpt", action="store_true", help="ckpt 파일 test 실행")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    cfg.ckpt = args.ckpt
    
    test_model(cfg)