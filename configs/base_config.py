from albumentations import Compose, Resize, HorizontalFlip, GridDropout, CoarseDropout
from albumentations.pytorch import ToTensorV2

class BaseConfig:
    def __init__(self):
        self.project_name = "SemanticSegmentation"
        self.run_name = "Unet++"
        self.image_root = "/data/ephemeral/home/data/train/DCM"
        self.label_root = "/data/ephemeral/home/data/train/outputs_json"
        self.batch_size = 16
        self.num_workers = 4
        self.input_size = 512
        self.max_epoch = 100
        self.valid_split = 0.2
        self.valid_interval = 1
        self.checkpoint_dir = "./checkpoints"
        self.seed = 42
        self.lr = 0.0001
        self.train_transforms = Compose([
            Resize(height=self.input_size, width=self.input_size),
            HorizontalFlip(p=0.5),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            GridDropout(ratio=0.5, unit_size_min=32, unit_size_max=64, p=0.5),
            ToTensorV2(),
        ])
        self.valid_transforms = Compose([
            Resize(height=self.input_size, width=self.input_size),
            ToTensorV2(),
        ])

    def get_transforms(self, mode="train"):
        if mode == "train":
            return self.train_transforms
        elif mode == "valid":
            return self.valid_transforms
        else:
            raise ValueError(f"Unsupported mode: {mode}")