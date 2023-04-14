import random
from pprint import pformat
from dataclasses import dataclass
from functools import partial

import torch
from lightning.fabric import Fabric
from torchmetrics.classification import MulticlassF1Score

from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch import optim, nn

from ..utils import build_dataloader
from ..processors.classification import Classifier
from .kie import build_model


@dataclass
class Metrics:
    best_f1: float = 0.0
    val_loss: float = 999999

    def __str__(self):
        return " - ".join(f"{k} {v}" for k, v in vars(self).items())

    def update(self, best_f1=None, val_loss=None):
        updated = False

        # Best F1 score
        if best_f1 is not None:
            updated = updated or self.best_f1 <= best_f1
            self.best_f1 = max(best_f1, self.best_f1)

        if val_loss is not None:
            updated = updated or (self.val_loss >= val_loss)
            self.val_loss = min(best_f1, self.best_f1)

        return updated


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
        # TODO: this
        self.processor = Classifier(
            tokenizer=tokenizer,
            label_map=model_config.task['classes'])

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
            num_batches=len(self.train_loader)
        )

        # Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.validate_every,
            num_training_steps=self.config.training_time
        )

        # Evaluation and logging
        self.metrics = Metrics()
        self.score = MulticlassF1Score(
            len(model_config.task['classes']) + 1,
            ignore_index=-100
        )

        self.latest_weight_path = model_config['latest_weight_path']
        self.best_weight_path = model_config['best_weight_path']
        self.log_dir = model_config['log_dir']

    def fit(self):
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        train_loader = self.fabric.setup_dataloaders(self.train_loader)
        criterion = self.criterion
        lr_scheduler = self.lr_scheduler

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
            classes = batch['classes']
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            polygon_ids=polygon_ids)
            loss = criterion(outputs.transpose(-1, 1), classes)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            loss = loss.item()

            if step % validate_every == 0:
                updated = self.validate()
                state_dict = model.state_dict()
                if updated:
                    torch.save(state_dict, self.best_weight_path)
                torch.save(state_dict, self.latest_weight_path)
                tqdm.write(f"Step {step}/{total_steps} - {self.metrics}")

            if step % print_every == 0:
                tqdm.write(f"Step {step}/{total_steps} - {self.metrics}")

            lr = optimizer.param_groups[0]['lr']
            pbar.set_description(f"Loss {loss:.2e} - Lr {lr:.2e}")

    @torch.no_grad()
    def validate(self):
        model = self.fabric.setup(self.model.eval())
        val_loader = self.fabric.setup_dataloaders(self.val_loader)

        tqdm.write("Validation")
        nb = len(val_loader)
        d_idx = random.choice(range(nb))

        f1s = []
        losses = []

        pbar = tqdm(val_loader, "Validate", dynamic_ncols=True)
        for idx, batch in enumerate(pbar):
            input_ids = batch['input_ids']
            polygon_ids = batch['polygon_ids']
            token_mapping = batch['token_mapping']
            class_ids = batch['classes']
            outputs = model(input_ids=input_ids,
                            polygon_ids=polygon_ids)
            loss = self.criterion(outputs.transpose(-1, 1), classes)
            self.score.to(input_ids.device)
            f1 = self.score(outputs.argmax(dim=-1), class_ids)
            f1s.append(f1.cpu().item())
            losses.append(loss.cpu().item())

            if d_idx == idx:
                class_logits = torch.softmax(outputs, dim=-1)
                for (i, m, c, l) in zip(input_ids, token_mapping, class_ids, class_logits):
                    gt = self.processor.decode(i, m, class_ids=c)
                    pr = self.processor.decode(i, m, class_logits=l)
                tqdm.write("PR: " + pformat(pr))
                tqdm.write("GT: " + pformat(gt))

        mean_f1 = sum(f1s) / len(f1s)
        val_loss = sum(f1s) / len(f1s)
        self.metrics.update(val_loss=val_loss)
        updated = self.metrics.update(best_f1=mean_f1)
        return updated
