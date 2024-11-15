from .base_config import BaseConfig

class LinknetConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "Linknet"