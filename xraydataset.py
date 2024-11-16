import numpy as np
import os
import cv2

import torch
import json

from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

from constants import CLASSES, CLASS2IND

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

class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, label_files=None, transforms=None):
        self.image_files = image_files
        self.label_files = label_files  # Optional for test set without labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        label_path = self.label_files[item] if self.label_files else None

        # 이미지 로드
        image = cv2.imread(image_path)
        if len(image.shape) == 2:  # Grayscale 처리
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image / 255.0

        # 레이블 로드
        if label_path:
            label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
            label = np.zeros(label_shape, dtype=np.uint8)
            with open(label_path, "r") as f:
                annotations = json.load(f)["annotations"]
            for ann in annotations:
                c = ann["label"]
                class_ind = CLASS2IND[c]
                points = np.array(ann["points"], dtype=np.int32)
                cv2.fillPoly(label[..., class_ind], [points], 1)
        else:
            label = np.zeros((len(CLASSES), *image.shape[:2]), dtype=np.uint8)

        # 증강 적용
        if self.transforms:
            inputs = {"image": image, "mask": label} if self.label_files else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.label_files else label

        # 디버깅: 증강 후 형태 확인
        print(f"[DEBUG] Transformed image shape: {image.shape}")
        if label is not None:
            print(f"[DEBUG] Transformed label shape: {label.shape}")

        # 배열에서 Tensor로 변환
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # HWC -> CHW
        label = torch.from_numpy(label.transpose(2, 0, 1)).float() if self.label_files else None

        return (image, label) if label is not None else image