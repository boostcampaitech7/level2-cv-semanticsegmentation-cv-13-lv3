from .base_config import BaseConfig

class UPerNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "UPerNet"
        self.encoder_name = "resnet50"
        self.encoder_weights = "imagenet"