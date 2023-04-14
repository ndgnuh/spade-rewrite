import yaml
from os import path
from typing import *
from dataclasses import dataclass, field


def _resolve_time(s, num_batches):
    if s.endswith("steps"):
        return int(s.replace("steps", "").replace(" ", ""))
    elif s.endswith("epochs"):
        return int(s.replace("epochs", "").replace(" ", "")) * num_batches
    else:
        raise ValueError(
            f"time string must ends with 'steps' or 'epochs', eg '100 epochs'"
        )


@dataclass
class ModelConfig:
    name: str
    task: Dict
    backbone: Dict
    head: Dict = field(default_factory=dict)
    weights: Optional[str] = None

    def __post_init__(self):
        assert self.task.get("name", None) in (
            "classification", "masked-modelling")

        if self.weights is not None and not self.weights.startswith("http"):
            assert path.isfile(self.weights)

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def best_weight_path(self):
        return path.join("storage", "weights", f"{self.name}.pt")

    @property
    def latest_weight_path(self):
        return path.join("storage", "weights", f"{self.name}-latest.pt")

    @property
    def log_dir(self):
        return path.join("storage", "logs", self.name)


@dataclass
class TrainConfig:
    training_time: str
    train_data: str
    val_data: str
    lr: float
    validate_every: str
    batch_size: Optional[int] = 1
    num_workers: Optional[int] = 0
    print_every: Optional[str] = None

    # Sub config
    lightning_config: Optional[Dict] = field(default_factory=dict)
    model_config: Optional[Dict] = None
    dataloader: Optional[Dict] = field(default_factory=dict)

    # Resolve flags
    _resolved: bool = False

    def __post_init__(self):
        # Type conversion
        self.lr = float(self.lr)

        # Check if file exists
        assert path.exists(self.train_data)
        assert path.exists(self.val_data)

    def __getitem__(self, key):
        return getattr(self, key)

    # Convert time strings to time steps
    # so that the trainer can use them
    def resolve(self, num_batches):
        if self._resolved:
            return

        # Convert everything to steps
        self.training_time = _resolve_time(self.training_time, num_batches)
        self.validate_every = _resolve_time(self.validate_every, num_batches)

        # Get the default print_every
        if self.print_every is not None:
            self.print_every = _resolve_time(self.print_every, num_batches)
        else:
            self.print_every = max(self.validate_every // 5, 1)

        # Set the resolved flag to true
        self._resolved = True


def validate_training_config(config):
    mandatory = [
        "train_data",
        "valid_data",
        "lr",
        "training_time",
        "validate_every",
        "batch_size",
    ]

    optional = ["num_workers"]
    for key in mandatory:
        assert key in config.keys(), f"Missing key '{key}' in model config"

    # Warn for unknown key
    for key in config.keys():
        if key not in mandatory:
            print(f"Unknown key '{key}'")


def validate_model_config(config):
    mandatory = ["type", "replace_word_embeddings"]
    for key in mandatory:
        assert key in config, f"Missing key '{key}' in model config"

    # Warn for unknown key
    for key in config.keys():
        if key not in mandatory:
            print(f"Unknown key '{key}'")


def _read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def read_model_config(config_file):
    config = _read_config(config_file)
    if "name" not in config:
        name = path.splitext(path.basename(config_file))[0]
        config["name"] = name

    config = ModelConfig(**config)
    return config


def read_training_config(config_file):
    config = _read_config(config_file)
    return TrainConfig(**config)
