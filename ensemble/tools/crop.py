import os
import json
import sys
import numpy as np
#from xraydataset_edit import XRayDataset
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'lightningmodule'))
from utils import get_sorted_files_by_type
from constants import TRAIN_DATA_DIR

# 라벨 파일에서 크롭 정보 추출
def extract_crop_info(label_path):
    """
    JSON 라벨 파일에서 클래스별 크롭 정보를 추출합니다.
    Args:
        label_path (str): 라벨 파일 경로
    Returns:
        dict: 클래스별 크롭 정보 (min/max 좌표)
    """
    with open(label_path, "r") as f:
        data = json.load(f)
        annotations = data.get("annotations", [])
        
        if not annotations:
            print(f"[WARNING] {label_path}: 'annotations' 데이터가 비어 있습니다.")
            return {}
        
        class_minmax = {}
        for ann in annotations:
            label = ann["label"]
            points = np.array(ann["points"])
            min_coords = np.min(points, axis=0).tolist()
            max_coords = np.max(points, axis=0).tolist()
            class_minmax[label] = {"min": min_coords, "max": max_coords}
        
        return class_minmax

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