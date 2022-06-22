from torch import nn
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
from . import graph_stuff as G
from . import data
import torch
import json
from argparse import Namespace
import numpy as np


def tensorize(x):
    return torch.as_tensor(np.array(x))


class Transpose(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b

    def forward(self, x):
        return x.transpose(self.dim_a, self.dim_b)


class RelationTagger(nn.Module):
    def __init__(
        self,
        n_relations,
        backbone_hidden_size,
        head_hidden_size,
        head_p_dropout=0.1,
    ):
        super().__init__()
        hidden_size = backbone_hidden_size
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.W_0 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden):
        h = self.W_h(hidden)
        d = self.W_d(hidden)
        score = torch.einsum("bih,bjh->bij", h, self.W_0(d)).unsqueeze(0)
        return score


def partially_from_pretrained(config, model_name, **kwargs):
    pretrain = AutoModel.from_pretrained(model_name, **kwargs)
    model = type(pretrain)(config)
    pretrain_sd = pretrain.state_dict()
    for (k, v) in model.named_parameters():
        if k not in pretrain_sd:
            continue
        if pretrain_sd[k].data.shape == v.shape:
            v.data = pretrain_sd[k].data

    return model


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def poly_to_box(poly):
    x = [p[0] for p in poly]
    y = [p[1] for p in poly]
    return [min(x), min(y), max(x), max(y)]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def gen_classifier_label(rel, text_tokens, fields):
    # rel_tokens = data.expand_rel_s(rel, tokenizer, texts, coords, fields)

    classification = [None for _ in text_tokens]
    for (i, j) in zip(*np.where(rel)):
        if i < len(fields):
            classification[j] = i

    for _ in text_tokens:
        for (i, j) in zip(*np.where(rel)):
            i = i - len(fields)
            if i < 0:
                continue
            if classification[i] is not None:
                classification[j] = classification[i]

    other_label = len(fields)
    special_label = len(fields) + 1
    classification = [other_label if l is None else l for l in classification]
    return AttrDict(
        other_label=other_label,
        special_label=special_label,
        classification=classification,
    )


def parse_input(
    image,
    words,
    actual_boxes,
    tokenizer,
    config,
    label,
    fields,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
):
    width, height = image.size
    boxes = [normalize_box(b, width, height) for b in actual_boxes]

    tokens = []
    token_boxes = []
    actual_bboxes = []  # we use an extra b because actual_boxes is already used
    token_actual_boxes = []
    are_box_first_tokens = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))
        are_box_first_tokens.extend([1] + [0] * (len(word_tokens) - 1))
    if label is not None:
        rel_s = tensorize(label[0])
        rel_g = tensorize(label[1])

        # CLASSIFICATION LABEL
        token_map = G.map_token(tokenizer, words, offset=len(fields))
        token_rel_s = G.expand(rel_s, token_map, lh2ft=True, in_tail=True, in_head=True)
        classifier_label = gen_classifier_label(token_rel_s, tokens, fields)
        other_label = classifier_label.other_label
        special_label = classifier_label.other_label
        classification = classifier_label.classification

        # SPAN LABEL
        token_rel_g = G.expand(rel_g, token_map, lh2ft=True, in_tail=True, in_head=True)
        # span_label = G.graph2span_classes(token_rel_g)
        span_label = G.graph2span(token_rel_s, token_rel_g)
    else:
        other_label = len(fields)
        special_label = len(fields)
        classification = [0 for _ in tokens]
        span_label = [0 for _ in tokens]

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    real_max_len = config.max_position_embeddings - special_tokens_count
    if len(tokens) > real_max_len:
        tokens = tokens[:real_max_len]
        token_boxes = token_boxes[:real_max_len]
        actual_bboxes = actual_bboxes[:real_max_len]
        token_actual_boxes = token_actual_boxes[:real_max_len]
        are_box_first_tokens = are_box_first_tokens[:real_max_len]
        classification = classification[:real_max_len]
        span_label = span_label[:real_max_len]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]
    are_box_first_tokens += [1]
    classification += [special_label]
    span_label += [4]

    segment_ids = [0] * len(tokens)

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids
    are_box_first_tokens = [2] + are_box_first_tokens
    classification = [special_label] + classification
    span_label = [4] + span_label
    # span_label = [l + 1 if l > 0 else l for l in span_label]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = config.max_position_embeddings - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length
    are_box_first_tokens += [3] * padding_length
    classification += [special_label] * padding_length
    span_label += [4] * padding_length
    # print(len(classification), len(input_ids))

    assert len(input_ids) == config.max_position_embeddings
    assert len(input_mask) == config.max_position_embeddings
    assert len(segment_ids) == config.max_position_embeddings
    assert len(token_boxes) == config.max_position_embeddings
    assert len(token_actual_boxes) == config.max_position_embeddings
    assert len(are_box_first_tokens) == config.max_position_embeddings
    assert len(classification) == config.max_position_embeddings
    assert len(span_label) == config.max_position_embeddings
    # print(set(span_label))
    # import sys

    return {
        "text_tokens": tokens,
        "input_ids": tensorize(input_ids).unsqueeze(0),
        "attention_mask": tensorize(input_mask).unsqueeze(0),
        "token_type_ids": tensorize(segment_ids).unsqueeze(0),
        "bbox": torch.clamp(tensorize(token_boxes), 0, 1000).unsqueeze(0),
        "actual_bbox": tensorize(token_actual_boxes).unsqueeze(0),
        "labels_c": tensorize(classification).unsqueeze(0),
        "labels_s": tensorize(span_label).unsqueeze(0),
        # "stc_labels": stc_labels.unsqueeze(0),
        # "labels": tensorize(labels).unsqueeze(0),
        "are_box_first_tokens": tensorize(are_box_first_tokens).unsqueeze(0),
    }


def batch_parse_input(tokenizer, config, batch_data):
    batch = []
    text_tokens = []
    for d in batch_data:
        texts = d["text"]
        actual_boxes = [poly_to_box(b) for b in d["coord"]]
        image = Namespace(size=(d["img_sz"]["width"], d["img_sz"]["height"]))
        label = d["label"]
        fields = d["fields"]
        features = parse_input(
            image, texts, actual_boxes, tokenizer, config, label=label, fields=fields
        )
        text_tokens.append(features.pop("text_tokens"))
        batch.append(features)

    batch_features = {}
    for key in batch[0]:
        batch_features[key] = torch.cat([b[key] for b in batch], dim=0)

    batch_features["text_tokens"] = text_tokens

    return batch_features


class RelationExtractor(nn.Module):
    def __init__(
        self,
        n_relations,
        backbone_hidden_size,
        head_hidden_size,
        head_p_dropout=0.1,
    ):
        super().__init__()

        self.n_relations = n_relations
        self.backbone_hidden_size = backbone_hidden_size
        self.head_hidden_size = head_hidden_size
        self.head_p_dropout = head_p_dropout

        self.drop = nn.Dropout(head_p_dropout)
        self.q_net = nn.Linear(
            self.backbone_hidden_size, self.n_relations * self.head_hidden_size
        )

        self.k_net = nn.Linear(
            self.backbone_hidden_size, self.n_relations * self.head_hidden_size
        )

        self.dummy_node = nn.Parameter(torch.Tensor(1, self.backbone_hidden_size))
        nn.init.normal_(self.dummy_node)

    def forward(self, h_q, h_k):
        h_q = self.q_net(self.drop(h_q))

        # dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, h_k.size(1), 1)
        # h_k = torch.cat([h_k, dummy_vec], axis=0)
        h_k = self.k_net(self.drop(h_k))

        head_q = h_q.view(
            h_q.size(0), h_q.size(1), self.n_relations, self.head_hidden_size
        )
        head_k = h_k.view(
            h_k.size(0), h_k.size(1), self.n_relations, self.head_hidden_size
        )

        relation_score = torch.einsum("ibnd,jbnd->nbij", (head_q, head_k))

        return relation_score


class RelationExtractor(nn.Module):
    def __init__(
        self,
        n_relations,
        backbone_hidden_size,
        head_hidden_size,
        head_p_dropout=0.1,
    ):
        super().__init__()

        self.n_relations = n_relations
        self.backbone_hidden_size = backbone_hidden_size
        self.head_hidden_size = head_hidden_size
        self.head_p_dropout = head_p_dropout

        self.drop = nn.Dropout(head_p_dropout)
        self.q_net = nn.Linear(
            self.backbone_hidden_size, self.n_relations * self.head_hidden_size
        )

        self.k_net = nn.Linear(
            self.backbone_hidden_size, self.n_relations * self.head_hidden_size
        )

        self.dummy_node = nn.Parameter(torch.Tensor(1, self.backbone_hidden_size))
        nn.init.normal_(self.dummy_node)

    def forward(self, h_q, h_k):
        h_q = self.q_net(self.drop(h_q))

        # dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, h_k.size(1), 1)
        # h_k = torch.cat([h_k, dummy_vec], axis=0)
        h_k = self.k_net(self.drop(h_k))

        head_q = h_q.view(h_q.size(0), h_q.size(1), self.head_hidden_size)
        head_k = h_k.view(h_k.size(0), h_k.size(1), self.head_hidden_size)

        relation_score = torch.einsum("ibd,jbd->bij", (head_q, head_k))

        return relation_score


@dataclass
class SpadeOutput:
    itc_outputs: torch.Tensor
    stc_outputs: torch.Tensor
    attention_mask: torch.Tensor
    loss: Optional[torch.Tensor] = None


def hybrid_layoutlm(config_layoutlm, config_bert, layoutlm, bert, **kwargs):
    if layoutlm == bert:
        return AutoModel.from_pretrained(layoutlm, **kwargs)

    layoutlm = partially_from_pretrained(config_layoutlm, layoutlm, **kwargs)
    bert = partially_from_pretrained(config_bert, bert, **kwargs)
    layoutlm.embeddings.word_embeddings = bert.embeddings.word_embeddings
    layoutlm.embeddings.position_embeddings = bert.embeddings.position_embeddings
    return layoutlm


class LayoutLMSpade(nn.Module):
    def __init__(self, config, num_labels, layoutlm=None, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        if layoutlm is None:
            layoutlm = "microsoft/layoutlm-base-cased"
        self.backbone = AutoModel.from_pretrained(layoutlm, local_files_only=True)
        bert = AutoModel.from_pretrained("vinai/phobert-base", local_files_only=True)
        self.backbone.embeddings.word_embeddings = bert.embeddings.word_embeddings
        self.backbone.embeddings.position_embeddings = (
            bert.embeddings.position_embeddings
        )

        self.config = config = self.backbone.config
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classify = nn.Linear(config.hidden_size, num_labels)
        self.num_span_labels = 5
        self.span_classify = nn.Linear(config.hidden_size, self.num_span_labels)

        self.max_position_embeddings = bert.config.max_position_embeddings

    def forward(self, input_ids, bbox, attention_mask, labels_c=None, labels_s=None):
        out = self.backbone(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask
        )
        lhs = out.last_hidden_state

        logits_c = self.classify(self.dropout(lhs))
        hidden = torch.einsum("bih,bin->bih", lhs, logits_c)
        logits_s = self.span_classify(self.dropout(hidden))

        losses = [0, 0]
        if labels_c is not None:
            lf = nn.CrossEntropyLoss()
            losses[0] = lf(logits_c.view(-1, self.num_labels), labels_c.view(-1))

        if labels_s is not None:
            lf = nn.CrossEntropyLoss()
            losses[1] = lf(
                logits_s.view(-1, self.num_span_labels),
                labels_s.view(-1),
            )

        return Namespace(loss=sum(losses), logits_c=logits_c, logits_s=logits_s)


class SpadeDataset(Dataset):
    def __init__(self, tokenizer, config, jsonl, test_mode=False):
        super().__init__()
        with open(jsonl) as f:
            data = [json.loads(line) for line in f.readlines()]

        self.raw = data
        self.fields = data[0]["fields"]
        if test_mode:
            for d in data:
                d["label"] = None
        self.nfields = len(self.fields)
        self._cached_length = len(data)
        self.features = batch_parse_input(tokenizer, config, data)
        self.text_tokens = self.features.pop("text_tokens")

    def __len__(self):
        return self._cached_length

    def __getitem__(self, idx):
        return BatchEncoding(
            {key: self.features[key][idx] for key in self.features.keys()}
        )
