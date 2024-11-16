import os
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from augmentation import Cutout, Grid_dropout

class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, transforms=None, clahe=False, copypaste=False):
        """
        Args:
            image_root (str): 이미지 경로
            label_root (str): 레이블 경로
            transforms (albumentations.Compose): 데이터 변환
            clahe (bool): CLAHE 적용 여부
            copypaste (bool): Copy-Paste augmentation 적용 여부
        """
        self.image_root = image_root
        self.label_root = label_root
        self.transforms = transforms
        self.clahe = clahe
        self.copypaste = copypaste
        self.image_files = self._get_files(image_root)
        self.label_files = self._get_files(label_root)

    def _get_files(self, root_dir):
        return sorted([os.path.join(root_dir, fname) for fname in os.listdir(root_dir)])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.label_files[idx], cv2.IMREAD_GRAYSCALE)

        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        if self.transforms:
            augmented = self.transforms(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        return image, label

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, 
                 image_root, 
                 label_root, 
                 batch_size=16, 
                 num_workers=4, 
                 transforms=None, 
                 valid_split=0.2, 
                 clahe=False, 
                 copypaste=False):
        """
        PyTorch Lightning DataModule을 사용하는 데이터 관리 클래스.
        Args:
            image_root (str): 이미지 데이터 경로
            label_root (str): 레이블 데이터 경로
            batch_size (int): 배치 크기
            num_workers (int): DataLoader의 worker 수
            transforms (dict): Augmentation 설정
            valid_split (float): 검증 데이터 비율
            clahe (bool): CLAHE 적용 여부
            copypaste (bool): Copy-Paste augmentation 적용 여부
        """
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
        """
        데이터셋 초기화 및 Train/Validation split 처리
        """
        train_transforms = self._create_transforms(self.transforms['train'])
        valid_transforms = self._create_transforms(self.transforms['valid'])

        full_dataset = XRayDataset(
            image_root=self.image_root,
            label_root=self.label_root,
            transforms=train_transforms,  
            clahe=self.clahe,
            copypaste=self.copypaste
        )

        val_size = int(len(full_dataset) * self.valid_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        self.val_dataset.dataset.transforms = valid_transforms

    def train_dataloader(self):
        """
        학습 데이터 로더 반환
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """
        검증 데이터 로더 반환
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)

    def _create_transforms(self, transforms_config):
        """
        YAML 설정에 따라 Albumentations 변환을 생성.
        Args:
            transforms_config (list): Augmentation 설정 리스트
        Returns:
            A.Compose: Albumentations 변환 객체
        """
        transform_list = []
        for transform in transforms_config:
            t_type = transform['type']
            params = transform.get('params', {})
            if t_type == "cutout":
                transform_list.append(Cutout(**params))
            elif t_type == "grid_dropout":
                transform_list.append(Grid_dropout(**params))
            # 다른 augmentation 추가
        return A.Compose(transform_list)