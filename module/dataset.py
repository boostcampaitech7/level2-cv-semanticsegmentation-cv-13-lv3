import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, img_paths, mask_paths=None, labels=None, transforms=None, concat=False, resize=384):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transforms = transforms
        self.concat = concat
        self.resize = resize

    def __len__(self):
        return len(self.img_paths)

    def _preprocess(self, image):
        # 이미지 전처리 (Normalize 등)
        image = cv2.resize(image, (self.resize, self.resize))
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        return image

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = self._preprocess(image)

        if self.mask_paths:
            mask_path = self.mask_paths[index]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize, self.resize))
            return image, mask

        if self.labels:
            label = self.labels[index]
            return image, label

        return image


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, mask_dir=None, batch_size=16, num_workers=4, test_size=0.2):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size

    def setup(self, stage=None):
        img_paths = sorted(glob(os.path.join(self.img_dir, "*.png")))
        mask_paths = sorted(glob(os.path.join(self.mask_dir, "*.png"))) if self.mask_dir else None

        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            img_paths, mask_paths, test_size=self.test_size, random_state=42
        )

        self.train_dataset = CustomDataset(train_imgs, train_masks, transforms=A.Compose([
            A.Resize(384, 384),
            A.Normalize(),
            ToTensorV2()
        ]))
        self.val_dataset = CustomDataset(val_imgs, val_masks, transforms=A.Compose([
            A.Resize(384, 384),
            A.Normalize(),
            ToTensorV2()
        ]))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)