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

def get_config(model_name):
    if model_name in CONFIG_MAP:
        return CONFIG_MAP[model_name]()
    raise ValueError(f"Unsupported model name: {model_name}")

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