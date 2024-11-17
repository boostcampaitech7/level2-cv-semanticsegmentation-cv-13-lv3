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
    if not pngs or not jsons:
        raise ValueError(f"No image or label files found. Images: {len(pngs)}, Labels: {len(jsons)}")

    assert valid_idx <= K

    _filenames = np.array(pngs)
    _labelnames = np.array(jsons)

    groups = [os.path.dirname(fname) for fname in _filenames]
    ys = [0 for fname in _filenames]

    gkf = GroupKFold(n_splits=min(K, len(_filenames)))

    train_datalist, valid_datalist = dict(filenames=[], labelnames=[]), dict(filenames=[], labelnames=[])

    for idx, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
        if idx + 1 == valid_idx:
            valid_datalist['filenames'] += list(_filenames[y])
            valid_datalist['labelnames'] += list(_labelnames[y])
        else:
            train_datalist['filenames'] += list(_filenames[y])
            train_datalist['labelnames'] += list(_labelnames[y])

    return train_datalist, valid_datalist

class XRayDataset(Dataset):
    def __init__(self, image_files, label_files=None, transforms=None):
        """
        image_files : list of image file paths
        label_files : list of label file paths (None for test sets)
        """
        self.image_files = image_files
        self.label_files = label_files  # Optional for test set without labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, item):
        image_path = self.image_files[item]
        image_name = os.path.basename(image_path)

        image = cv2.imread(image_path)
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
            # No labels for test set
            label = np.zeros((len(CLASSES), *image.shape[:2]), dtype=np.uint8)
        
        if self.transforms:
            inputs = {"image": image, "mask": label} if self.label_files else {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"] if self.label_files else label

        output_size = (128, 128)  # 모델 출력 크기
        resized_label = np.zeros((len(CLASSES), *output_size), dtype=np.uint8)
        for i in range(len(CLASSES)):
            resized_label[i] = cv2.resize(label[..., i], output_size, interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        label = torch.from_numpy(resized_label).float() if self.label_files else None

        return (image_name, image, label) if label is not None else (image_name, image)