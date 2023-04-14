from dataclasses import dataclass
from typing import *
from collections import defaultdict


@dataclass
class Classifier:
    tokenizer: Any
    label_map: List[str]

    def __post_init__(self):
        n = len(self.label_map)
        self.label_map = {k: label for k, label in enumerate(self.label_map)}
        self.label_map[-100] = 'other'
        self.label_map[n] = 'other'

    def decode(self,
               input_ids,
               token_mapping,
               class_ids=None,
               class_logits=None):
        assert class_ids is not None or class_logits is not None
        if class_ids is None:
            scores, class_ids = class_logits.max(dim=-1)
        else:
            scores = 1

        outputs = defaultdict(list)
        for i in token_mapping.unique():
            mask = (token_mapping == i)
            text = self.tokenizer.decode(input_ids[mask].tolist())
            class_id = class_ids[mask].mode()[0].item()
            outputs[class_id].append(text)

        outputs = {self.label_map[k]: " ".join(v)
                   for k, v in outputs.items()}
        outputs.pop("other")
        return outputs
