from transformers import AutoModel, AutoTokenizer
from typing import Optional
from torch import nn

from .bros import BrosModel, BrosTokenizer

def _replace_layer(
    source: nn.Module,
    target: nn.Module,
    path: str
):
    pass

def build_model(
    config,
    replace_word_embeddings: Optional[str] = None,
    bros_pretrain = "base",
):
    model = KIE(
        num_classes=len(config["classes"]) + 1,
        head_size=config["head_size"],
        bros_pretrain=bros_pretrain
    )
    if replace_word_embeddings is not None:
        rep = AutoModel.from_pretrained(replace_word_embeddings)
        model.backbone.embeddings.word_embeddings = rep.embeddings.word_embeddings
        tokenizer = AutoTokenizer.from_pretrained(replace_word_embeddings)
    else:
        tokenizer = BrosTokenizer.from_pretrained(f"naver-clova-ocr/bros-{bros_pretrain}-uncased")

    return model, tokenizer

class KIE(nn.Module):
    def __init__(
        self,
        head_size: int,
        num_classes: int,
        bros_pretrain: str
    ):
        super().__init__()
        assert bros_pretrain in ["base", "large"]
        self.backbone = BrosModel.from_pretrained(f"naver-clova-ocr/bros-{bros_pretrain}-uncased")
        self.config = self.backbone.config

        self.projection = nn.Sequential(
            nn.Linear(self.config.hidden_size, head_size),
            nn.GELU()
        )

        self.classify = nn.Linear(head_size, num_classes, bias=False)

    def forward(self, input_ids, polygon_ids, attention_mask=None):
        hiddens = self.backbone(
            input_ids=input_ids,
            bbox=polygon_ids,
            attention_mask=attention_mask
        )

        x = self.projection(hiddens.last_hidden_state)
        cls = self.classify(x)
        return cls

