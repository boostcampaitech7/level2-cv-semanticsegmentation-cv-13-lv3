from .unet_config import UnetConfig
from .unetpp_config import UnetPlusPlusConfig
from .manet_config import MAnetConfig
from .linknet_config import LinknetConfig
from .fpn_config import FPNConfig
from .pspnet_config import PSPNetConfig
from .pan_config import PANConfig
from .deeplabv3_config import DeepLabV3Config
from .deeplabv3plus_config import DeepLabV3PlusConfig
from .upernet_config import UPerNetConfig

CONFIG_MAP = {
    "Unet": UnetConfig,
    "Unet++": UnetPlusPlusConfig,
    "MAnet": MAnetConfig,
    "Linknet": LinknetConfig,
    "FPN": FPNConfig,
    "PSPNet": PSPNetConfig,
    "PAN": PANConfig,
    "DeepLabV3": DeepLabV3Config,
    "DeepLabV3+": DeepLabV3PlusConfig,
    "UPerNet": UPerNetConfig,
}

def get_config(model_name):
    """
    모델 이름에 해당하는 구성 설정 클래스를 반환합니다.

    Args:
        model_name (str): 모델 이름

    Returns:
        Config: 해당 모델의 설정 클래스

    Raises:
        ValueError: 지원하지 않는 모델 이름일 경우
    """
    if model_name in CONFIG_MAP:
        return CONFIG_MAP[model_name]()
    raise ValueError(f"Unsupported model name: {model_name}")