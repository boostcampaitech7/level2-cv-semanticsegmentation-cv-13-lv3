from .base_config import BaseConfig

class UPerNetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "UPerNet"
        self.encoder_name = "resnet50"  # Transformer-based backbone
        self.encoder_weights = "imagenet"  # Pre-trained weights
        self.classes = 29