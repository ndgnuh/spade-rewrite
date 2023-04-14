from lightning.fabric import Fabric
from .kie import build_model


class Trainer:
    def __init__(self, config):
        model_config = config['model']
        self.fabric = Fabric(config['fabric'])
        self.model = build_model(model_config)
