from configs.unet_config import UNetConfig
from configs.unetpp_config import UNetPlusPlusConfig
from configs.manet_config import MAnetConfig
from configs.linknet_config import LinknetConfig
from configs.fpn_config import FPNConfig
from configs.pspnet_config import PSPNetConfig
from configs.pan_config import PANConfig
from configs.deeplabv3_config import DeepLabV3Config
from configs.deeplabv3plus_config import DeepLabV3PlusConfig
from configs.upernet_config import UPerNetConfig
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
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

def build_model(model_name, encoder_name, encoder_weights, num_classes):
    model_name_map = {
        "Unet++": "UnetPlusPlus",  # Add a mapping for "Unet++"
        "UNet": "Unet",
        "MAnet": "MAnet",
        "Linknet": "Linknet",
        "FPN": "FPN",
        "PSPNet": "PSPNet",
        "PAN": "PAN",
        "DeepLabV3": "DeepLabV3",
        "DeepLabV3Plus": "DeepLabV3Plus",
        "UPerNet": "UPerNet",
    }
    mapped_model_name = model_name_map.get(model_name)
    if mapped_model_name and mapped_model_name in smp.__dict__:
        return smp.__dict__[mapped_model_name](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
