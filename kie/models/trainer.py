from lightning.fabric import Fabric
from .kie import build_model


class Trainer:
    def __init__(self, config):
        model_config = config.model_config
        self.fabric = Fabric(**config.lightning_config)
        self.model = build_model(model_config)


    def fit(self):
        pass
