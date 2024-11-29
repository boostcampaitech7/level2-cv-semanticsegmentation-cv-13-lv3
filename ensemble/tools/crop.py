import os
import json
import sys
import numpy as np
#from xraydataset_edit import XRayDataset
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'lightningmodule'))
from utils import get_sorted_files_by_type, decode_rle_to_mask
from constants import TRAIN_DATA_DIR

import pandas as pd

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
PALM_CLASSES =  ['Hamate', 'Scaphoid', 'Lunate', 'Trapezium', 'Capitate', 'Triquetrum', 'Trapezoid', 'Pisiform']

PALM_CLASS2IND = {v: i for i, v in enumerate(PALM_CLASSES)}
PALM_IND2CLASS = {v: k for k, v in PALM_CLASS2IND.items()}

# 라벨 파일에서 크롭 정보 추출
def extract_crop_info(csv_path):
    """
    JSON 라벨 파일에서 클래스별 크롭 정보를 추출합니다.
    Args:
        label_path (str): 라벨 파일 경로
    Returns:
        dict: 클래스별 크롭 정보 (min/max 좌표)
    """
    df = pd.read_csv(csv_path)

    crop_info = dict()

    # 그룹화하여 처리
    grouped = df.groupby('image_name')
    for idx, (image_name, group) in enumerate(grouped):
        crop_info[image_name] = dict()
        masks = []
        for _, row in group.iterrows():
            classname = row['class']
            if classname in PALM_CLASSES:
                rle = row['rle']
                if isinstance(rle, str):
                    mask = decode_rle_to_mask(rle, 2048, 2048)
                    masks.append(mask)

        combined_mask = np.logical_or.reduce(masks)

        y_indices, x_indices = np.where(combined_mask == True)
        # 최소 및 최대 좌표 계산
        min_x, max_x = x_indices.min(), x_indices.max()
        min_y, max_y = y_indices.min(), y_indices.max()

        crop_info[image_name]['min'] = (min_x, min_y)
        crop_info[image_name]['max'] = (max_x, max_y)

    return crop_info

# 크롭 정보 저장 함수
def save_crop_info(image_files, label_files, save_path):
    """
    이미지와 라벨 파일을 기반으로 크롭 정보를 추출하여 저장합니다.
    Args:
        image_files (list): 이미지 파일 리스트
        label_files (list): 라벨 파일 리스트
        save_path (str): 저장할 JSON 파일 경로
    """
    assert len(image_files) == len(label_files), "이미지 파일과 라벨 파일의 개수가 일치하지 않습니다."
    
    crop_info = {}
    for image_path, label_path in zip(image_files, label_files):
        image_name = os.path.basename(image_path)
        class_minmax = extract_crop_info(label_path)
        
        if not class_minmax:
            print(f"[WARNING] {image_name}: 크롭 정보가 비어 있습니다.")
            crop_info[image_name] = {}
        else:
            crop_info[image_name] = class_minmax
        
        # 진행 상태 출력
        if len(crop_info) % 50 == 0:
            print(f"[INFO] {len(crop_info)}/{len(image_files)} 이미지 처리 중...")
    
    # JSON 저장
    with open(save_path, "w") as f:
        json.dump(crop_info, f, indent=4)
    print(f"[완료] 크롭 정보가 {save_path}에 저장되었습니다.")

# 경로 정의
image_root = os.path.join(TRAIN_DATA_DIR, 'DCM')
label_root = os.path.join(TRAIN_DATA_DIR, 'outputs_json')

# 이미지와 라벨 파일 로드
image_files = get_sorted_files_by_type(image_root, 'png')
label_files = get_sorted_files_by_type(label_root, 'json')

# JSON 저장 경로
save_path = os.path.join(TRAIN_DATA_DIR, "crop_info.json")

# 크롭 정보 저장 실행
save_crop_info(image_files, label_files, save_path)