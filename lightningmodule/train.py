# torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xraydataset import XRayDataset, split_data
from snapmix import SnapMixDataset
from utils import get_sorted_files_by_type, set_seed, Gsheet_param
from constants import PALM_TRAIN_DATA_DIR
from argparse import ArgumentParser, Namespace
import albumentations as A
import os
import torch
from model_lightning import SegmentationModel
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from augmentation import load_transforms
from test import test_model  # 테스트 함수 임포트
from loss import *
import numpy as np


# 체크포인트 콜백 클래스 : ckpt에 bestEp 저장 + 학습 종료시 torch.save로 pt 저장
class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _format_checkpoint_name(self, filename, metrics, auto_insert_metric_name=False):
        # metrics에서 epoch 값 가져오기
        epoch_num = f"{metrics['epoch']:02d}" if 'epoch' in metrics else "unknown"

        # 파일 이름 형식 지정 (에폭 번호만 포함, val/dice 제거)
        return f"{self.filename}-bestEp_{epoch_num}"
    
    def on_train_end(self, trainer, pl_module):
        if not os.path.exists(self.dirpath):  
            os.makedirs(self.dirpath, exist_ok=True)  
        # 학습이 모두 끝났을 때 전체 모델 저장
        model_path = os.path.join(self.dirpath, f"{self.filename}-final.pt")
        torch.save(pl_module, model_path)
        print(f"Final model saved at: {model_path}")

        super().on_train_end(trainer, pl_module)


# 모델 학습과 검증을 수행하는 함수
def train_model(args):
    args_dict = OmegaConf.to_container(args, resolve=True)
    run_name = args_dict.pop('run_name', None)
    project_name = args_dict.pop('project_name', None)
    seed_everything(args.seed)
    set_seed(args.seed)

    # resume 체크포인트 경로 지정
    if args.resume_checkpoint_suffix == None:
        resume_checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}.ckpt")
    else:
        resume_checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_file}{args.resume_checkpoint_suffix}.ckpt")
        
    if args.resume:
        # Resume 설정 확인
        if os.path.exists(resume_checkpoint_path):
            print(f"Resume : <{resume_checkpoint_path}> 체크포인트에서 학습 재개")
        else:
            raise FileNotFoundError(f"Resume : 체크포인트가 존재하지 않음 <{resume_checkpoint_path}>")
    else:
        resume_checkpoint_path = None
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
    image_root = os.path.join(PALM_TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(PALM_TRAIN_DATA_DIR, 'outputs_json')
    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')
    
    transforms = load_transforms(args)
    
    print(args.cp_args)
    
    train_dataset = XRayDataset(
        image_files=np.array(pngs),
        label_files=jsons,
        transforms=transforms,
        use_cp=args.use_cp,
        cp_args=args.cp_args
    )   
    
    # valid_dataset = XRayDataset(
    #     image_files=valid_files['filenames'],
    #     label_files=valid_files['labelnames'],
    #     transforms=transforms,
    # )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
      
    # # 주의: validation data는 이미지 크기가 크기 때문에 `num_wokers`는 커지면 메모리 에러가 발생할 수 있습니다.
    # valid_loader = DataLoader(
    #     dataset=valid_dataset, 
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=7,
    #     drop_last=False
    # )


    criterion = calc_dice_loss()
    seg_model = SegmentationModel(
        criterion=criterion,
        learning_rate=args.lr,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weight=args.encoder_weight
    )
    
    # 체크포인트 콜백 : dice 기준 상위 k개
    checkpoint_callback_best = CustomModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.checkpoint_file}",
        monitor='val/dice',
        mode='max',
        save_top_k=3
    )

    # 체크포인트 콜백 : resume 보조 마지막 학습 체크포인트 저장
    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.checkpoint_file + "-latest",
        monitor=None,
        save_top_k=1,
        every_n_epochs=1  # 매 에폭마다 저장
    )

    trainer = Trainer(
        logger=wandb_logger,
        log_every_n_steps=5,
        max_epochs=args.max_epoch,
        check_val_every_n_epoch=args.valid_interval,
        callbacks=[checkpoint_callback_best, checkpoint_callback_latest,],
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if args.amp else 32,
    )

    # 학습 시작
    trainer.fit(seg_model, 
                train_dataloaders=train_loader, 
                # val_dataloaders=valid_loader,
                ckpt_path=resume_checkpoint_path if args.resume else None  # 체크포인트 경로 전달
                )
    
    # 학습 종료 후 테스트 수행
    if args.auto_eval:
        print("Train 완료 -> Test 시작!")
        test_model(args)  # 테스트 함수 호출

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--resume", action="store_true", help="resume으로 실행할 건지")
    parser.add_argument("--wandb_id", type=str, default=None, help="resume 할 때 WandB에서 기존 실험에 이어서 기록하게 wandb id")
    parser.add_argument("--auto_eval", action="store_true", help="학습 끝나고 자동으로 test 실행")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
        
    cfg.resume = args.resume
    cfg.wandb_id = args.wandb_id
    cfg.auto_eval = args.auto_eval
    
    train_model(cfg)
    Gsheet_param(cfg)
