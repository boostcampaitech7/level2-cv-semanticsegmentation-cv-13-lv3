import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from sklearn.model_selection import GroupKFold
import os
import cv2
import json
import numpy as np
import random
from torchvision import transforms

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

def do_clahe(image):
    """Apply CLAHE for contrast enhancement."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None, clahe=False, copypaste=False, k=3):
        # 파일 로드 및 정렬
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=label_root)
            for root, _dirs, files in os.walk(label_root)
            for fname in files if os.path.splitext(fname)[1].lower() == ".json"
        }
        pngs = sorted(pngs)
        jsons = sorted(jsons)

        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)

        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0 for fname in _filenames]

        gkf = GroupKFold(n_splits=5)
        dataset_no = 2
        filenames, labelnames = [], []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i != dataset_no:
                    filenames += list(_filenames[y])
                    labelnames += list(_labelnames[y])
            else:
                if i == dataset_no:
                    filenames = list(_filenames[y])
                    labelnames = list(_labelnames[y])
        self.filenames = filenames
        self.labelnames = labelnames
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        self.clahe = clahe
        self.copypaste = copypaste
        self.k = k

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_name = self.filenames[index]
        image_path = os.path.join(self.image_root, image_name)
        image = cv2.imread(image_path)
        if self.clahe:
            image = do_clahe(image)
        image = image / 255.0

        label_name = self.labelnames[index]
        label_path = os.path.join(self.label_root, label_name)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES),)
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

        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"]

        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class XRayDataModule(pl.LightningDataModule):
    def __init__(self, image_root, label_root, batch_size=16, num_workers=4, valid_split=0.2, transforms=None, clahe=False, copypaste=False):
        super().__init__()
        self.image_root = image_root
        self.label_root = label_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_split = valid_split
        self.transforms = transforms
        self.clahe = clahe
        self.copypaste = copypaste

    def setup(self, stage=None):
        dataset = XRayDataset(
            image_root=self.image_root,
            label_root=self.label_root,
            is_train=True,
            transforms=self.transforms,
            clahe=self.clahe,
            copypaste=self.copypaste
        )
        val_size = int(len(dataset) * self.valid_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)