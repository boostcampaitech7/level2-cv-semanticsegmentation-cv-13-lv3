from .base_config import BaseConfig

class DeepLabV3PlusConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "DeepLabV3+"