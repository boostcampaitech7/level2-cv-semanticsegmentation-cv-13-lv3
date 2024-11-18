import os
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from model_lightning import SegmentationModel  # 학습된 모델 클래스
from utils.utils import get_sorted_files_by_type  # 파일 정렬 함수
from constants import TEST_DATA_DIR

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def preprocess_image(image_path, input_size):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0) 

def infer(model, device, image_path, input_size):
    model.eval()
    image_tensor = preprocess_image(image_path, input_size).to(device)
    with torch.no_grad():
        output = model(image_tensor)['out']
    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    return pred_mask


def main():
    # 설정
    input_size = 512  # 모델 입력 크기
    checkpoint_path = 'path_to_trained_model.pth'  # 학습된 모델 경로
    output_csv = 'output.csv'  # 결과 저장 파일
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = SegmentationModel.load_from_checkpoint(checkpoint_path)
    model.to(device)

    # 테스트 이미지 파일 목록 가져오기
    image_root = os.path.join(TEST_DATA_DIR, 'DCM')
    image_files = get_sorted_files_by_type(image_root, 'png')

    # 결과 저장을 위한 리스트
    results = []

    # 각 이미지에 대해 추론 수행
    for image_file in image_files:
        image_path = os.path.join(image_root, image_file)
        pred_mask = infer(model, device, image_path, input_size)
        rle = rle_encode(pred_mask)
        results.append({'filename': image_file, 'rle': rle})

    # 결과를 CSV 파일로 저장
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')

if __name__ == '__main__':
    main()