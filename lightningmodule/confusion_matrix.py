# torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from xraydataset import XRayDataset, split_data
from utils.utils import get_sorted_files_by_type, set_seed
from constants import TRAIN_DATA_DIR, CLASSES
from argparse import ArgumentParser
import albumentations as A
import os
import torch
from model_lightning import SegmentationModel
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    # y_true: [N, C, H, W], y_pred: [N, C, H, W]
    y_true = y_true.cuda()
    y_pred = y_pred.cuda()
    
    data_size = y_true.size(0)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=y_true.device)
    
    for n in range(data_size):
        # 각 클래스의 Calculate confusion matrix
        for i in range(num_classes):
            for j in range(num_classes):
                true_i = y_true[n, i].flatten()  # b번째 배치의 i번째 클래스
                pred_j = y_pred[n, j].flatten()  # b번째 배치의 j번째 클래스
                
                # intersection (TP: True Positive)
                intersection = torch.sum(true_i * pred_j)
                
                # 실제 해당 클래스의 전체 픽셀 수
                total_true_pixels = torch.sum(true_i)
                
                # 비율 계산 (TP / Total True)
                ratio = intersection / (total_true_pixels + 1e-6)
                confusion_matrix[i, j] += ratio.item()
    
    return confusion_matrix

def save_confusion_matrix(confusion_matrix, classes):

    plt.figure(figsize=(15, 12))
    
    # heatmap 시각화
    sns.heatmap(confusion_matrix.cpu().numpy(), 
                annot=True,
                fmt='.1f',  # 소수점 3자리까지 표시
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Label이 잘리지 않도록 layout 조정
    plt.tight_layout()
    
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")
    
def decode_rle_to_mask(rle, height, width):
    if not isinstance(rle, str):
        return np.zeros(height * width, dtype=np.uint8)
    
    # numpy 배열로 한번에 변환    
    pairs = np.array(list(map(int, rle.strip().split())))
    starts = pairs[::2] - 1  # 1-based to 0-based indexing
    lengths = pairs[1::2]
    
    mask = np.zeros(height * width, dtype=np.uint8)
    
    # 벡터화된 인덱싱 사용
    ends = starts + lengths
    for start, end in zip(starts, ends):
        mask[start:end] = 1
        
    return mask.reshape(height, width)

def csv_to_imgs(csv_file, args):
    df = pd.read_csv(csv_file)
    num_images = len(df) // 29
    origin_size = 2048
    
    # 미리 전체 배열 할당
    imgs = np.zeros((num_images, 29, args.input_size, args.input_size), dtype=np.uint8)
    
    # 이미지 리사이즈 transform 미리 생성
    resize_transform = A.Resize(args.input_size, args.input_size)
    
    for idx in range(num_images):
        image_data = df.iloc[idx*29:(idx*29)+29]
        for class_idx, row in enumerate(image_data.itertuples()):
            if isinstance(row.rle, str):
                mask = decode_rle_to_mask(row.rle, origin_size, origin_size)
                mask = resize_transform(image=mask)['image']
                imgs[idx, class_idx] = mask
    
    return imgs

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
        transforms=A.Resize(args.input_size, args.input_size)
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    
    label_dataset = XRayDataset(
        image_files=valid_files['filenames'],
        label_files=valid_files['labelnames'],
        transforms=A.Resize(args.input_size, args.input_size)
    )
    
    label_loader = DataLoader(
        dataset=label_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    
    # 모델 및 데이터 로더 설정
    criterion = nn.BCEWithLogitsLoss()
    seg_model = SegmentationModel(
        criterion=criterion,
        learning_rate=args.lr,
        architecture=args.architecture,
        encoder_name=args.encoder_name,
        encoder_weight=args.encoder_weight
    )
    
    # Trainer 설정
    trainer = Trainer(
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if args.amp else 32,
    )

    # validation 실행
    trainer.test(seg_model, dataloaders=valid_loader, ckpt_path=checkpoint_path)
 
    # CSV 파일 읽기
    imgs = torch.from_numpy(csv_to_imgs("./output.csv", args)).float()  # 이미지 이름은 무시
    
    # 라벨 수집
    labels = torch.cat([batch[2] for batch in label_loader], dim=0)

    # Confusion matrix 계산
    confusion_matrix = calculate_confusion_matrix(labels, imgs, len(CLASSES))
    save_confusion_matrix(confusion_matrix, CLASSES)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    
    confusion_matrix(cfg)