import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lightningmodule'))
from model_lightning import SegmentationModel
from model_lightning_palm import SegmentationModel_palm
from utils import encode_mask_to_rle, decode_rle_to_mask, label2rgb
from xraydataset import XRayDataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mmsegmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SAM2-UNet'))

import json
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
PALM_CLASSES =  ['Hamate', 'Scaphoid', 'Lunate', 'Trapezium', 'Capitate', 'Triquetrum', 'Trapezoid', 'Pisiform']

# 크롭 정보를 로드하는 함수
def load_crop_info(crop_info_path):
    """
    crop_info.json 파일을 로드하여 딕셔너리로 반환합니다.
    Args:
        crop_info_path (str): crop_info.json 파일 경로
    Returns:
        dict: 크롭 정보를 담고 있는 딕셔너리
    """
    with open(crop_info_path, 'r') as f:
        crop_info = json.load(f)
    return crop_info

# 데이터셋 클래스 정의
class EnsembleDataset(Dataset):
    def __init__(self, image_root, cfg):
        """
        소프트 보팅 앙상블을 위한 데이터셋 클래스.
        Args:
            image_root (str): 이미지가 저장된 루트 경로
            cfg (dict): 설정 파일
        """
        self.image_root = image_root
        self.image_files = self._get_all_image_files(image_root)
        self.cfg = cfg
        
    def _get_all_image_files(self, root_dir):
        """
        하위 폴더를 포함하여 모든 이미지 파일 경로를 반환합니다.
        Args:
            root_dir (str): 루트 디렉토리

        Returns:
            list: 이미지 파일 경로 목록
        """
        image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return sorted(image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        주어진 인덱스의 이미지를 로드하고 반환합니다.
        """
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_root, image_name)
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"[ERROR] 이미지 파일을 로드할 수 없습니다: {image_path}")
        
        image = image.astype(np.float32) / 255.0
        return {"images": image, "image_names": image_name}

    def collate_fn(self, batch):
        """
        배치 데이터를 생성하는 함수입니다.
        """
        images = [data["images"] for data in batch]
        image_names = [data["image_names"] for data in batch]
        images = torch.tensor(images).permute(0, 3, 1, 2)
        return {"images": images, "image_names": image_names}

# 크롭 데이터를 복원하는 함수
def restore_cropped_outputs(crop_info, cropped_outputs, original_shape):
    """
    크롭된 출력을 원본 이미지 크기로 복원합니다.
    Args:
        crop_info (dict): 크롭 정보 (min, max 좌표 포함)
        cropped_outputs (np.ndarray): 크롭된 출력
        original_shape (tuple): 원본 이미지의 크기
    Returns:
        np.ndarray: 원본 크기로 복원된 출력
    """
    restored = np.zeros(original_shape, dtype=np.float32)
    for class_idx, coords in crop_info.items():
        if coords:
            min_y, min_x = coords['min']
            max_y, max_x = coords['max']
            restored[min_y:max_y, min_x:max_x] = cropped_outputs[class_idx]
    return restored

# 결과를 저장하는 함수
def save_results(cfg, filename_and_class, rles):
    """
    추론 결과를 CSV 파일로 저장합니다.
    Args:
        cfg (dict): 설정 파일
        filename_and_class (list): 파일 이름 및 클래스 정보 리스트
        rles (list): RLE로 인코딩된 세그멘테이션 마스크 리스트
    """
    import pandas as pd
    save_path = os.path.join(cfg.save_dir, cfg.output_name)
    
    # 디렉토리 없으면 생성
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 결과 데이터프레임 생성
    results = pd.DataFrame({
        "filename_class": filename_and_class,
        "rle": rles
    })
    results.to_csv(save_path, index=False)
    print(f"결과가 {save_path}에 저장되었습니다.")

# 소프트 보팅 함수
def soft_voting(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Palm 모델 로드
    palm_models = []
    if cfg.palm_model_paths:
        palm_models = [
            torch.load(path).to(device).eval()
            for path in cfg.palm_model_paths
        ]
        print(f"{len(palm_models)}개의 Palm 모델이 로드되었습니다.")
    else:
        print("Palm 모델이 제공되지 않았습니다. Palm 모델 추론을 건너뜁니다.")

    # 일반 모델 로드
    general_models = []
    if cfg.smp_model_paths:
        general_models = [
            torch.load(path).to(device).eval()
            for path in cfg.smp_model_paths
        ]
        print(f"{len(general_models)}개의 일반 모델이 로드되었습니다.")
    else:
        print("일반 모델이 제공되지 않았습니다. 일반 모델 추론을 건너뜁니다.")

    # 모델이 하나도 없는 경우 예외 처리
    if not palm_models and not general_models:
        raise ValueError("모델이 제공되지 않았습니다. `palm_model_paths`와 `smp_model_paths`가 모두 비어 있습니다.")

    # 크롭 정보 로드
    crop_info = load_crop_info(cfg.crop_info_path)

    # 데이터 로드
    dataset = EnsembleDataset(cfg.image_root, cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.collate_fn
    )

    palm_weight = float(cfg.get("palm_weight", 0.5))
    general_weight = float(cfg.get("smp_weight", 0.5))
    print(f"[INFO] Palm Weight: {palm_weight}, SMP Weight: {general_weight}")

    # 가중치 합 검증
    if not abs(palm_weight + general_weight - 1.0) < 1e-6:
        raise ValueError("Palm 모델과 일반 모델의 가중치 합이 1이 아닙니다.")

    # `prediction_save`가 False인 경우
    if not cfg.get("prediction_save", True):
        print("[INFO] Prediction 저장을 건너뛰고 바로 앙상블을 진행합니다.")
        filename_and_class = []
        rles = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images, image_names = batch["images"].to(device), batch["image_names"]

                # Palm 모델 추론 및 복원
                palm_outputs_restored = {cls: torch.zeros((len(images), 2048, 2048)).to(device) for cls in PALM_CLASSES}
                for palm_model in palm_models:
                    palm_outputs = palm_model(images)
                    for i, image_name in enumerate(image_names):
                        crop_infos = crop_info.get(image_name, {})
                        if crop_infos:
                            restored = restore_cropped_outputs(crop_infos, palm_outputs[i].cpu().numpy(), (2048, 2048))
                            for class_name, restored_output in restored.items():
                                palm_outputs_restored[class_name][i] += torch.tensor(restored_output).to(device)

                if palm_models:
                    for class_name in palm_outputs_restored:
                        palm_outputs_restored[class_name] /= len(palm_models)

                # 일반 모델 추론
                general_outputs = {cls: torch.zeros((len(images), 2048, 2048)).to(device) for cls in CLASSES}
                for general_model in general_models:
                    outputs = general_model(images)
                    for idx, class_name in enumerate(CLASSES):
                        general_outputs[class_name] += outputs[:, idx, :, :]

                if general_models:
                    for class_name in general_outputs:
                        general_outputs[class_name] /= len(general_models)

                # 클래스별 결합 및 Threshold 적용
                for class_idx, class_name in enumerate(CLASSES):
                    if class_name in PALM_CLASSES:
                        palm_output = palm_outputs_restored[class_name]
                        combined_output = (
                            palm_weight * palm_output
                            + general_weight * general_outputs[class_name]
                        )
                    else:
                        combined_output = general_outputs[class_name]

                    threshold = cfg.class_thresholds[class_idx]
                    binary_output = (combined_output > threshold).float()

                    for img_idx, segm in enumerate(binary_output):
                        rle = encode_mask_to_rle(segm.cpu().numpy())
                        rles.append(rle)
                        filename_and_class.append(f"{class_name}_{image_names[img_idx]}")

        # 최종 결과 저장
        save_results(cfg, filename_and_class, rles)
        return  # 저장 없이 바로 종료

    # 저장 관련 로직 (prediction_save가 True일 때만 실행)
    if cfg.get("prediction_save", True):
        # Save or Save_Load 모드
        if cfg.mode in ["save", "save_load"]:
            print("[INFO] Prediction 저장을 시작합니다!")
            os.makedirs(cfg.save_dir, exist_ok=True)  # 저장 디렉토리 생성
            for batch_idx, batch in enumerate(data_loader):
                images, image_names = batch["images"].to(device), batch["image_names"]

                # Palm 모델 추론 및 복원
                palm_outputs_restored = {cls: torch.zeros((len(images), 2048, 2048)).to(device) for cls in PALM_CLASSES}
                for palm_model in palm_models:
                    palm_outputs = palm_model(images)
                    for i, image_name in enumerate(image_names):
                        crop_infos = crop_info.get(image_name, {})
                        if crop_infos:
                            restored = restore_cropped_outputs(crop_infos, palm_outputs[i].cpu().numpy(), (2048, 2048))
                            for class_name, restored_output in restored.items():
                                palm_outputs_restored[class_name][i] += torch.tensor(restored_output).to(device)

                if palm_models:
                    for class_name in palm_outputs_restored:
                        palm_outputs_restored[class_name] /= len(palm_models)

                # 일반 모델 추론
                general_outputs = {cls: torch.zeros((len(images), 2048, 2048)).to(device) for cls in CLASSES}
                for general_model in general_models:
                    outputs = general_model(images)
                    for idx, class_name in enumerate(CLASSES):
                        general_outputs[class_name] += outputs[:, idx, :, :]

                if general_models:
                    for class_name in general_outputs:
                        general_outputs[class_name] /= len(general_models)

                # 클래스별 결합 및 저장
                for class_idx, class_name in enumerate(CLASSES):
                    if class_name in PALM_CLASSES:
                        palm_output = palm_outputs_restored[class_name]
                        combined_output = (
                            palm_weight * palm_output
                            + general_weight * general_outputs[class_name]
                        )
                    else:
                        combined_output = general_outputs[class_name]

                    # 저장 경로 설정
                    save_path = os.path.join(
                        cfg.save_dir,
                        f"{cfg.prediction_file}_batch{batch_idx}_class{class_idx}.pt"
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 디렉토리 생성
                    torch.save(combined_output.cpu(), save_path)
                    print(f"Batch {batch_idx}, Class {class_name} predictions saved at {save_path}")

        # Load or Save_Load 모드
        if cfg.mode in ["load", "save_load"]:
            print("[INFO] Prediction 불러오기를 시작합니다!")
            all_combined_outputs = []
            for batch_idx in range(len(data_loader)):
                for class_idx, class_name in enumerate(CLASSES):
                    batch_load_path = os.path.join(
                        cfg.save_dir,
                        f"{cfg.prediction_file}_batch{batch_idx}_class{class_idx}.pt"
                    )
                    class_output = torch.load(batch_load_path)
                    all_combined_outputs.append((class_name, class_output))

            # Threshold 적용 및 최종 결과 처리
            filename_and_class = []
            rles = []
            for class_name, class_output in all_combined_outputs:
                threshold = cfg.class_thresholds[CLASSES.index(class_name)]
                binary_output = (class_output > threshold).float()

                for image_idx, segm in enumerate(binary_output):
                    rle = encode_mask_to_rle(segm.cpu().numpy())
                    rles.append(rle)
                    filename_and_class.append(f"{class_name}_{dataset.image_files[image_idx]}")

            # 최종 결과 저장
            save_results(cfg, filename_and_class, rles)


# 메인 함수
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/soft_voting_setting.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    soft_voting(cfg)