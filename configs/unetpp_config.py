from .base_config import BaseConfig

class UNetPlusPlusConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "UnetPlusPlus"
        self.encoder_name = "resnet34"
        self.encoder_weights = "imagenet"
        self.classes = 29