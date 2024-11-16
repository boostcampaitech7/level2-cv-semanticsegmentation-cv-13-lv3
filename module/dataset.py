import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None, augments=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        self.is_train = is_train
        self.augments = augments  # 추가된 부분
        self.images = sorted([os.path.join(image_root, img) for img in os.listdir(image_root)])
        self.labels = sorted([os.path.join(label_root, lbl) for lbl in os.listdir(label_root)])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels[idx]
        
        image = cv2.imread(img_path)
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

        # Albumentations transformations
        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Custom augmentations
        if self.augments:
            for augment in self.augments:
                if callable(augment):
                    image, label = augment(image, label)

        return image, label

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, image_root, label_root, batch_size=16, num_workers=4, transforms=None, valid_split=0.2, clahe=False, copypaste=False):
        super().__init__()
        self.image_root = image_root
        self.label_root = label_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.valid_split = valid_split
        self.clahe = clahe
        self.copypaste = copypaste

    def setup(self, stage=None):
        dataset = XRayDataset(
            image_root=self.image_root,
            label_root=self.label_root,
            transforms=self.transforms,
            clahe=self.clahe,
            copypaste=self.copypaste
        )
        val_size = int(len(dataset) * self.valid_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)