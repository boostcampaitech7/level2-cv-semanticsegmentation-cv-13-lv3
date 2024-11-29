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
from model_lightning_palm import SegmentationModel_palm
from model_lightning_ensemble import SegmentationModel_ensemble
from lightning.pytorch import Trainer
import numpy as np
from omegaconf import OmegaConf

def get_models(args, path_list):
    models = []
    if args.pt:
        for path in path_list:
            pt_path = os.path.join(args.checkpoint_dir, f"{path}")
            print(pt_path)
            model = torch.load(pt_path)  # 전체 모델 저장된 경우
            models.append(model)    
            
    return models

# 테스트를 수행하는 함수
def test_model(args):
    if args.pt:
        if args.ensemble:
            palm_models = get_models(args, args.palm_model_paths)
            general_models = get_models(args, args.smp_model_paths)
            seg_model = SegmentationModel_ensemble(gt_csv=args.standard_csv_path, model_weights=args.model_weights, thresholds=args.class_thresholds)
            seg_model.set_model(palm_models, general_models)
        else:
            pt_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}-final.pt")
            model = torch.load(pt_path)  # 전체 모델 저장된 경우
            if args.palm:
                seg_model = SegmentationModel_palm(gt_csv=args.standard_csv_path)
                seg_model.set_model(model)
            else:
                seg_model = model
    else:
        # 모델 및 체크포인트 경로 설정
        if args.resume_checkpoint_suffix == None:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt")
        else:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}{args.resume_checkpoint_suffix}.ckpt")
        seg_model = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path, criterion=None, learning_rate=None)

    image_files = None
    if args.valid:
        image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
        label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

        pngs = get_sorted_files_by_type(image_root, 'png')
        jsons = get_sorted_files_by_type(label_root, 'json')

        _, valid_files = split_data(pngs, jsons)

        image_files = valid_files['filenames']
    else:
        #데이터 로드 및 테스트
        image_root = os.path.join(TEST_DATA_DIR, 'DCM')
        pngs = get_sorted_files_by_type(image_root, 'png')

        image_files = np.array(pngs)

    test_dataset = XRayDataset(image_files=np.array(pngs),
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
    parser.add_argument("--pt", action="store_true", help="pt 파일 test 실행")
    parser.add_argument("--palm", action="store_true", help="palm 모델 test 실행")
    parser.add_argument("--ensemble", action="store_true", help="ensemble 모델 test 실행")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    cfg.pt = args.pt
    
    test_model(cfg)