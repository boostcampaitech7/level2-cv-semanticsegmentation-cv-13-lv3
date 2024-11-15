from .base_config import BaseConfig

class PANConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "PAN"