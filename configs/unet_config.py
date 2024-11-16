from .base_config import BaseConfig

class UNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "UNet"
        self.encoder_name = "efficientnet-b3"  # Encoder backbone
        self.encoder_weights = "imagenet"  # Pre-trained weights
        self.classes = 29