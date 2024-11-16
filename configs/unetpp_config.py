from .base_config import BaseConfig

class UNetPlusPlusConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "Unet++"
        self.encoder_name = "resnet34"  # 사용할 Encoder
        self.encoder_weights = "imagenet"  # Pre-trained weights
        self.classes = 29