import torch
from lightning.fabric import Fabric

from tqdm import tqdm
from functools import partial
from torch import optim

from ..utils import build_dataloader
from .kie import build_model


def loop(loader, total_steps):
    count = 0
    while True:
        for batch in loader:
            if count == total_steps:
                return
            count = count + 1
            yield count, batch


class Trainer:
    def __init__(self, config):
        # Store
        model_config = config.model_config
        self.config = config
        self.model_config = model_config

        # Model
        self.fabric = Fabric(**config.lightning_config)
        self.model, tokenizer = build_model(model_config)

        # Make dataloaders
        _build_dataloader = partial(
            build_dataloader,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            **config.dataloader
        )
        self.train_loader = _build_dataloader(config.train_data)
        self.val_loader = _build_dataloader(config.val_data)

        # Resolve training scheduling
        config.resolve(
            num_batches = len(self.train_loader)
        )

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def fit(self):
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        train_loader = self.fabric.setup_dataloaders(self.train_loader)

        # Timing
        total_steps = self.config.training_time
        validate_every = self.config.validate_every
        print_every = self.config.print_every

        pbar = tqdm(
            loop(train_loader, total_steps),
            total=total_steps
        )
        for step, batch in pbar:
            input_ids = batch['input_ids']
            polygon_ids = batch['polygon_ids']
            ic(input_ids.shape, polygon_ids.shape)
            break

            if step % validate_every == 0:
                self.validate()

            if step % print_every == 0:
                tqdm.write("Print information")

            pbar.set_description(f"Training #{step}")

    @torch.no_grad()
    def validate(self):
        model = self.fabric.setup(self.model.eval())
        val_loader = self.fabric.setup_dataloaders(self.val_loader)

        tqdm.write("Validation")
