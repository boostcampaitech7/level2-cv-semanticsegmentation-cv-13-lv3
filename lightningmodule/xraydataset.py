from constants import CLASSES, CLASS2IND

import numpy as np
import os
import cv2
import random

import torch
import json

from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

import albumentations as A

def split_data(pngs, jsons, K=5, valid_idx=5):

    assert valid_idx <= K

    _filenames = np.array(pngs)
    _labelnames = np.array(jsons)

    # split train-valid
    # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
    # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
    # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
    groups = [os.path.dirname(fname) for fname in _filenames]

    # dummy label
    ys = [0 for fname in _filenames]

    # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
    # 5으로 설정하여 KFold를 수행합니다.
    gkf = GroupKFold(n_splits=K)

    train_datalist, valid_datalist = dict(filenames = [], labelnames = []), dict(filenames = [], labelnames = [])

    for idx, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        if idx+1 == valid_idx:
            valid_datalist['filenames'] += list(_filenames[y])
            valid_datalist['labelnames'] += list(_labelnames[y])
        else:
            train_datalist['filenames'] += list(_filenames[y])
            train_datalist['labelnames'] += list(_labelnames[y])

    return train_datalist, valid_datalist

class XRayDataset(Dataset):
    def __init__(self, image_files, label_files=None, transforms=None, use_cp=False, cp_args=None):
        """
        image_files : list of image file paths
        label_files : list of label file paths (None for test sets)
        """
        self.image_files = image_files
        self.label_files = label_files
        self.transforms = transforms
        
        self.use_cp = use_cp
        self.cp_args = cp_args
        
        self.mask_points = []
        self.class_minmax = []
    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        
        max = np.max(polygon_np, axis=0) # max_x, max_y
        min = np.min(polygon_np, axis=0) # min_x, min_y
        
        return {'max': max, 'min': min}
    
    def copypaste(self, image, label, cp_class):
        def is_overlap(rect1, rect2):
            max_x1, max_y1, min_x1, min_y1 = rect1
            max_x2, max_y2, min_x2, min_y2 = rect2
            
            if max_x1 < min_x2 or max_x2 < min_x1:
                return False
            if max_y1 < min_y2 or max_y2 < min_y1:
                return False
            
            return True

        cp_idx = CLASS2IND[cp_class]
        cp_info = self.class_minmax[cp_idx]
        x_dist = cp_info['max'][1] - cp_info['min'][1]
        y_dist = cp_info['max'][0] - cp_info['min'][0]
        
        cp_rect = 0, 0, 0, 0
        safe_count = 0
        pos_found = False
        
        # 이미지 크기
        img_height, img_width, _ = image.shape
        
        # 격자 기반 좌표 생성
        grid_size = 256  # 격자 크기 설정
        grid_x = list(range(0, img_width - x_dist, grid_size))
        grid_y = list(range(0, img_height - y_dist, grid_size))
        random.shuffle(grid_x)
        random.shuffle(grid_y)
        
        while not pos_found and safe_count < 1000:
            if not grid_x or not grid_y:
                break
            
            # 격자에서 랜덤 선택
            x_start = grid_x.pop(random.randint(0, len(grid_x) - 1))
            y_start = grid_y.pop(random.randint(0, len(grid_y) - 1))
            x_end, y_end = x_start + x_dist, y_start + y_dist
            
            cp_rect = x_end, y_end, x_start, y_start
            overlapped = False
            
            for classidx, _ in enumerate(self.class_minmax):
                class_info = self.class_minmax[classidx]
                class_rect = class_info['max'][1], class_info['max'][0], class_info['min'][1], class_info['min'][0]
                overlapped = is_overlap(cp_rect, class_rect)
                if overlapped:
                    break
            
            if not overlapped:
                pos_found = True
            
            safe_count += 1
        
        if pos_found:
            # 복사할 마스크와 이미지를 패치
            cropped_image = image[cp_info['min'][0]:cp_info['max'][0], cp_info['min'][1]:cp_info['max'][1]].copy()
            new_points = (np.array(self.mask_points[cp_idx]) - cp_info['min'] + (y_start, x_start)).astype(np.int32)
            
            new_max = (y_start + y_dist, x_start + x_dist)
            new_min = (y_start, x_start)
            
            # 이미지와 마스크 업데이트
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.fillPoly(mask, [new_points], (1, 1, 1))
            
            for channel in range(3):  # RGB 채널 모두 처리
                image[y_start:y_start + y_dist, x_start:x_start + x_dist, channel] = np.where(
                    mask[y_start:y_start + y_dist, x_start:x_start + x_dist, channel] == 1,
                    cropped_image[..., channel],
                    image[y_start:y_start + y_dist, x_start:x_start + x_dist, channel]
                )
            
            label_class = label[..., cp_idx].astype(np.uint8)
            cv2.fillPoly(label_class, [new_points], 1)
            label[..., cp_idx] = label_class
            
            self.class_minmax.append({'max': new_max, 'min': new_min})
        
        return image, label   
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, item):
        
        if self.use_cp:
            self.mask_points = [None]*29
            self.class_minmax = [None]*29
        
        image_path = self.image_files[item]
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path).astype(np.float32)
        if self.label_files:
            label_path = self.label_files[item]
            label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
            label = np.zeros(label_shape, dtype=np.uint8)
            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]
            for ann in annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"])
                class_label = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_label, [points], 1)
                label[..., class_ind] = class_label
                
                if self.use_cp:
                    self.mask_points[class_ind]=points
                    self.class_minmax[class_ind]=self.get_coord(points)
                
        else:
            # No labels for test set
            label = np.zeros((len(CLASSES), *image.shape[:2]), dtype=np.uint8)
           
        if self.use_cp:    
            for cp_class in self.cp_args:      
                image, label = self.copypaste(image, label, cp_class)
            
        if self.transforms:
            inputs = {"image": image, "mask": label} if self.label_files else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.label_files else label
            
        image = image / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label.transpose(2, 0, 1)).float() if self.label_files else None
        return (image_name, image, label) if label is not None else (image_name, image)