from .unet_config import UNetConfig
from .unetpp_config import UNetPlusPlusConfig
from .manet_config import MAnetConfig
from .linknet_config import LinknetConfig
from .fpn_config import FPNConfig
from .pspnet_config import PSPNetConfig
from .pan_config import PANConfig
from .deeplabv3_config import DeepLabV3Config
from .deeplabv3plus_config import DeepLabV3PlusConfig
from .upernet_config import UPerNetConfig
from albumentations import Compose, Resize, HorizontalFlip, GridDropout, CoarseDropout
import segmentation_models_pytorch as smp

CONFIG_MAP = {
    "UNet": UNetConfig,
    "Unet++": UNetPlusPlusConfig,
    "MAnet": MAnetConfig,
    "Linknet": LinknetConfig,
    "FPN": FPNConfig,
    "PSPNet": PSPNetConfig,
    "PAN": PANConfig,
    "DeepLabV3": DeepLabV3Config,
    "DeepLabV3Plus": DeepLabV3PlusConfig,
    "UPerNet": UPerNetConfig,
}

class Config:
    def __init__(self):
        self.model_name = "Unet++"
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
            Resize(height=512, width=512),
            HorizontalFlip(p=0.5),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
            GridDropout(ratio=0.5, unit_size_min=32, unit_size_max=64, p=0.5)
        ])
        self.valid_transforms = Compose([
            Resize(height=512, width=512)
        ])

def get_config():
    return Config()

def build_model(model_name, encoder_name, encoder_weights, num_classes):
    if model_name == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "UNet":
        return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "UNetPlusPlus":
        return smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "MAnet":
        return smp.MAnet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "Linknet":
        return smp.Linknet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "FPN":
        return smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "PSPNet":
        return smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "PAN":
        return smp.PAN(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    elif model_name == "UPerNet":
        return smp.UPerNet(encoder_name=encoder_name, encoder_weights=encoder_weights, classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")