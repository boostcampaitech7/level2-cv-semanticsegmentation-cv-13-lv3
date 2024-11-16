from albumentations import Compose, Resize, HorizontalFlip, GridDropout, CoarseDropout
from albumentations.pytorch import ToTensorV2
import albumentations as A

class BaseConfig:
    def __init__(self):
        self.seed = 42
        self.input_size = 512  # 높이와 너비를 32의 배수로 설정
        self.batch_size = 16
        self.num_workers = 4
        self.max_epoch = 100
        self.valid_interval = 1
        self.checkpoint_dir = "./checkpoints"
        self.project_name = "SemanticSegmentation"
        self.run_name = "Model_Run"
        self.lr = 0.0001
        self.image_root = "/data/ephemeral/home/data/train/DCM"
        self.label_root = "/data/ephemeral/home/data/train/outputs_json"

    def get_transforms(self, mode="train"):
        if mode == "train":
            return A.Compose([
                A.Resize(512, 512),  # 반드시 32의 배수로 설정
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        elif mode == "valid":
            return A.Compose([
                A.Resize(512, 512),  # 반드시 32의 배수로 설정
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            raise ValueError(f"Unsupported mode: {mode}")