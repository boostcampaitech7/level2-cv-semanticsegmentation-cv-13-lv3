from .base_config import BaseConfig

class UnetPlusPlusConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "Unet++"