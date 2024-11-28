import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from tools.ensemble import greedy_soup_weights
from model import load_model
from xraydataset import XRayDataset, split_data
from utils import dice_coef, get_sorted_files_by_type
from augmentation import load_transforms
from constants import TRAIN_DATA_DIR

if __name__ == "__main__":
    # Config 파일 로드
    config = OmegaConf.load("configs/greedy.yaml")

    # Checkpoint 디렉토리 설정
    checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    output_file = config.output_file

    checkpoint_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") or f.endswith(".ckpt")
    ]
    print(f"Found {len(checkpoint_files)} checkpoints: {checkpoint_files}")

    # Validation 데이터 준비
    print("Preparing validation dataset...")
    image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
    label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')
    pngs = get_sorted_files_by_type(image_root, 'png')
    jsons = get_sorted_files_by_type(label_root, 'json')

    # Train/Valid 분할
    train_data, valid_data = split_data(pngs, jsons, K=5, valid_idx=5)

    # Transforms 로드
    transforms = load_transforms(config)

    # Validation Dataset 및 DataLoader 생성
    valid_dataset = XRayDataset(
        image_files=valid_data['filenames'],
        label_files=valid_data['labelnames'],
        transforms=transforms
    )
    val_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False
    )

    # Base 모델 초기화
    print("Initializing base model...")
    base_model = load_model(
        architecture=config.model.architecture,
        encoder_name=config.model.encoder_name,
        encoder_weight=config.model.encoder_weight
    )

    # Greedy Soup 수행
    print("Performing Greedy Soup Ensemble...")
    combined_weights = greedy_soup_weights(
        paths=checkpoint_files,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 가중치를 모델에 적용
    base_model.load_state_dict(combined_weights, strict=False)

    # 최종 모델 저장
    save_path = os.path.join(checkpoint_dir, output_file)
    torch.save({"state_dict": base_model.state_dict()}, save_path)
    print(f"Greedy Soup model saved to {save_path}")