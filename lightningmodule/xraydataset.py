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
    def __init__(self, image_files, label_files=None, transforms=None, use_snapmix=False, beta=1.0):
        """
        image_files : list of image file paths
        label_files : list of label file paths (None for test sets)
        transforms : Albumentations transforms
        use_snapmix : bool, whether to apply SnapMix augmentation
        beta : float, beta parameter for SnapMix
        """
        self.image_files = image_files
        self.label_files = label_files  # Optional for test set without labels
        self.transforms = transforms
        self.use_snapmix = use_snapmix
        self.beta = beta

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        image_path = self.image_files[item]
        image_name = os.path.basename(image_path)

        image = cv2.imread(image_path).astype(np.float32)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image = image / 255.0

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
        else:
            label = np.zeros((len(CLASSES), *image.shape[:2]), dtype=np.uint8)

        if self.use_snapmix and len(self.image_files) > 1:
            next_index = (item + 1) % len(self.image_files)
            next_image_path = self.image_files[next_index]
            next_image = cv2.imread(next_image_path).astype(np.float32) / 255.0
            if next_image is None:
                raise ValueError(f"Could not load next image at {next_image_path}")

            if self.label_files:
                next_label_path = self.label_files[next_index]
                next_label_shape = tuple(next_image.shape[:2]) + (len(CLASSES), )
                next_label = np.zeros(next_label_shape, dtype=np.uint8)
                with open(next_label_path, "r") as f:
                    annotations = json.load(f)["annotations"]
                for ann in annotations:
                    c = ann["label"]
                    class_ind = CLASS2IND[c]
                    points = np.array(ann["points"])
                    class_label = np.zeros(next_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(class_label, [points], 1)
                    next_label[..., class_ind] = class_label
            else:
                next_label = np.zeros((len(CLASSES), *next_image.shape[:2]), dtype=np.uint8)

            lam = np.random.beta(self.beta, self.beta)
            H, W, _ = image.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx, cy = np.random.randint(W), np.random.randint(H)
            x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
            y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)

            image[y1:y2, x1:x2] = next_image[y1:y2, x1:x2]
            if label is not None:
                label[y1:y2, x1:x2] = next_label[y1:y2, x1:x2]

        if self.transforms:
            inputs = {"image": image, "mask": label} if self.label_files else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.label_files else label

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label.transpose(2, 0, 1)).float() if self.label_files else None

        return (image_name, image, label) if label is not None else (image_name, image)