import os
import torch
from lightning.pytorch import Trainer
from ensemble.ensemble import greedy_soup
from ensemble.soup_ens import load_checkpoint_weights, save_model_weights
from model_lightning import SegmentationModel  # 학습된 모델 클래스
from dataset import get_validation_loader  # Validation 데이터 로더 함수
from test import dice_score  # 평가 지표 (예: Dice Score)
from omegaconf import OmegaConf

if __name__ == "__main__":
    # Config 파일 로드
    config = OmegaConf.load("configs/base_config.yaml")

    # Validation 데이터 로더 준비
    val_loader = get_validation_loader(config)

    # Checkpoint 디렉토리 지정
    checkpoint_dir = "./checkpoints"

    # Checkpoint 파일 로드
    checkpoint_files = load_checkpoint_weights(checkpoint_dir)
    print(f"Found {len(checkpoint_files)} checkpoints: {checkpoint_files}")

    # Base 모델 초기화
    base_model = SegmentationModel(
        criterion=None,  # Loss는 필요 없음
        learning_rate=config.lr,
        architecture=config.architecture,
        encoder_name=config.encoder_name,
        encoder_weight=None
    )

    # Greedy Soup 앙상블 수행
    best_model = greedy_soup(
        model=base_model,
        paths=checkpoint_files,
        data=val_loader,
        metric=dice_score,  # 평가 메트릭 (Dice Score)
        device="cuda" if torch.cuda.is_available() else "cpu",
        update_greedy=True
    )

    # 최종 모델 저장
    save_model_weights(best_model, os.path.join(checkpoint_dir, "greedy_soup_best.pt"))