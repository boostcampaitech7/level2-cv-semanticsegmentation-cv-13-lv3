from .base_config import BaseConfig

class DeepLabV3PlusConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "DeepLabV3Plus"
        self.encoder_name = "mit_b0"  # Encoder backbone
        self.encoder_weights = "imagenet"  # Pre-trained weights
        self.classes = 29