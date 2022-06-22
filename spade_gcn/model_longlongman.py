import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
from . import graph_stuff as G
import torch
import json
from argparse import Namespace
from functools import reduce
from bros.bros import BrosModel, BrosConfig
from more_itertools import windowed, chunked


def partition_slice(total_lenght, segment_length, overlap):
    if overlap > 0:
        windows = windowed(range(total_lenght), segment_length, step=overlap)
    else:
        windows = chunked(range(total_lenght), segment_length)
    slices = []
    for w in windows:
        slices.append(slice(w[0], w[-1] + 1))
    return slices


def batch_consine_sim(batch):
    score = torch.einsum("bih,bjh->bij", batch, batch)
    inv_norm = 1 / torch.norm(batch, dim=-1)
    return torch.einsum("bij,bi,bj->bij", score, inv_norm, inv_norm)


def tensorize(x):
    try:
        return torch.tensor(np.array(x))
    except Exception:
        return torch.tensor(x)


def true_length(mask):
    batch = len(mask.shape) == 2
    if batch:
        return [true_length(m) for m in mask]

    return torch.where(mask == 1)[-1][-1]


class Transpose(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b

    def forward(self, x):
        return x.transpose(self.dim_a, self.dim_b)


class RelationTagger(nn.Module):
    def __init__(self, n_fields, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_size, hidden_size)
        self.tail = nn.Linear(hidden_size, hidden_size)
        self.field_embeddings = nn.Parameter(torch.rand(1, n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, enc):

        enc_head = self.head(enc)
        enc_tail = self.tail(enc)

        batch_size = enc_tail.size(0)
        field_embeddings = self.field_embeddings.expand(batch_size, -1, -1)
        enc_head = torch.cat([field_embeddings, enc_head], dim=1)

        score_0 = torch.matmul(enc_head, self.W_label_0(enc_tail).transpose(1, 2))
        score_1 = torch.matmul(enc_head, self.W_label_1(enc_tail).transpose(1, 2))

        score = torch.cat(
            [
                score_0.unsqueeze(1),
                score_1.unsqueeze(1),
            ],
            dim=1,
        )
        return score


def partially_from_pretrained(config, model_name, **kwargs):
    pretrain = BrosModel.from_pretrained(model_name, **kwargs)
    model = type(pretrain)(config)
    pretrain_sd = pretrain.state_dict()
    for (k, v) in model.named_parameters():
        if k not in pretrain_sd:
            continue
        if pretrain_sd[k].data.shape == v.shape:
            v.data = pretrain_sd[k].data

    return model


def normalize_box(box, width, height):
    bbox = tensorize(box)
    bbox[[0, 2, 4, 6]] = bbox[[0, 2, 4, 6]] / width
    bbox[[1, 3, 5, 7]] = bbox[[1, 3, 5, 7]] / height
    return bbox


def poly_to_box(poly):
    # x = [p[0] for p in poly]
    # y = [p[1] for p in poly]
    # return [min(x), min(y), max(x), max(y)]
    return reduce(lambda x, y: x + y, poly, [])


def parse_input(
    image,
    words,
    actual_boxes,
    tokenizer,
    config,
    label,
    fields,
    cls_token_box=[0] * 8,
    sep_token_box=None,
    pad_token_box=[0] * 8,
):
    if label is None:
        label = torch.zeros(
            2,
            len(fields) + config.max_position_embeddings,
            config.max_position_embeddings,
            dtype=torch.long,
        )
    width, height = image.size
    if sep_token_box is None:
        sep_token_box = [width, height] * 4
    boxes = actual_boxes
    label = tensorize(label)
    token_map = G.map_token(tokenizer, words, offset=len(fields))
    rel_s = tensorize(label[0])
    rel_g = tensorize(label[1])
    token_rel_s = G.expand(rel_s, token_map, lh2ft=True, in_tail=True, in_head=True)
    token_rel_g = G.expand(rel_g, token_map, fh2ft=True)
    label = torch.cat(
        [token_rel_s.unsqueeze(0), token_rel_g.unsqueeze(0)],
        dim=0,
    )

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

    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    true_length = config.max_position_embeddings - special_tokens_count
    if len(tokens) > true_length:
        tokens = tokens[:true_length]
        token_boxes = token_boxes[:true_length]
        token_actual_boxes = token_actual_boxes[:true_length]
        actual_bboxes = actual_bboxes[:true_length]
        are_box_first_tokens = are_box_first_tokens[:true_length]
        label = label[:, : (len(fields) + true_length), :true_length]

    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]
    are_box_first_tokens += [1]
    # use labels for auxilary result
    n, i, j = label.shape
    labels = torch.zeros((n, i + 1, j + 1), dtype=label.dtype)
    labels[:, :i, :j] = label
    label = labels

    segment_ids = [0] * len(tokens)

    # print("----")
    # edge_0 = []
    # for (i, j) in zip(*torch.where(token_rel_s)):
    #     l = [(fields + tokens)[i], tokens[j]]
    #     print(l)
    #     edge_0.append(" -> ".join(l))

    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids
    are_box_first_tokens = [2] + are_box_first_tokens
    # This is tricky because cls need to be inserted
    # after the labels
    nfields = len(fields)
    top_half = label[:, :nfields, :]
    bottom_half = label[:, nfields:, :]
    n, i, j = label.shape
    new_label = torch.zeros(n, i + 1, j + 1, dtype=label.dtype)
    new_label[:, :nfields, 1:] = top_half
    new_label[:, (nfields + 1) :, 1:] = bottom_half
    label = new_label

    #     print("----")
    #     print("AFter CLS")
    #     edge_1 = []
    #     for (i, j) in zip(*torch.where(label[0])):
    #         l = [(fields + tokens)[i], tokens[j]]
    #         print(l)
    #         edge_1.append(" -> ".join(l))
    #     print("----")

    #     for (i, j) in zip(edge_0, edge_1):
    #         print(i, j, i == j)

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

    assert len(input_ids) == config.max_position_embeddings
    assert len(input_mask) == config.max_position_embeddings
    assert len(segment_ids) == config.max_position_embeddings
    assert len(token_boxes) == config.max_position_embeddings
    assert len(token_actual_boxes) == config.max_position_embeddings
    assert len(are_box_first_tokens) == config.max_position_embeddings

    # Label parsing
    n, i, j = label.shape
    labels = torch.zeros(
        n,
        config.max_position_embeddings + len(fields),
        config.max_position_embeddings,
        dtype=label.dtype,
    )
    labels[:, :i, :j] = label
    label = labels

    # assert itc_labels.shape[0] == config.max_position_embeddings
    # assert stc_labels.shape[1] == config.max_position_embeddings
    # assert stc_labels.shape[2] == config.max_position_embeddings

    # labels = torch.cat(
    #     [torch.zeros(labels.shape[0], 1, config.max_position_embeddings), labels],
    #     dim=1,
    # )

    token_boxes = torch.tensor(token_boxes, dtype=torch.float).unsqueeze(0)
    token_boxes[:, :, [0, 2, 4, 6]] = token_boxes[:, :, [0, 2, 4, 6]] / width
    token_boxes[:, :, [1, 3, 5, 7]] = token_boxes[:, :, [1, 3, 5, 7]] / height

    # The unsqueezed dim is the batch dim for each type
    return {
        "text_tokens": tokens,
        "input_ids": tensorize(input_ids).unsqueeze(0),
        "attention_mask": tensorize(input_mask).unsqueeze(0),
        "token_type_ids": tensorize(segment_ids).unsqueeze(0),
        "bbox": token_boxes,
        # "actual_bbox": tensorize(token_actual_boxes).unsqueeze(0),
        # "itc_labels": itc_labels.unsqueeze(0),
        # "stc_labels": stc_labels.unsqueeze(0),
        "labels": tensorize(labels).unsqueeze(0),
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
    # for key in batch[0]:
    #     print(key, batch[0][key].shape)
    for key in batch[0]:
        batch_features[key] = torch.cat([b[key] for b in batch], dim=0)

    batch_features["text_tokens"] = text_tokens

    return batch_features


def hybrid_backbone(config_layoutlm, config_bert, layoutlm, bert, **kwargs):
    bert = partially_from_pretrained(config_bert, bert, **kwargs)
    layoutlm = partially_from_pretrained(config_layoutlm, layoutlm, **kwargs)
    layoutlm.embeddings.word_embeddings = bert.embeddings.word_embeddings
    layoutlm.embeddings.position_embeddings = bert.embeddings.position_embeddings
    return layoutlm


class LongLongManSpade(nn.Module):
    def __init__(self, fields, max_length=1154, overlap=64):
        super().__init__()
        # BACKBONE
        config = BrosConfig.from_pretrained("naver-clova-ocr/bros-base-uncased")
        self.backbone = partially_from_pretrained(
            config,
            "naver-clova-ocr/bros-base-uncased",
            local_files_only=True,
            # num_hidden_layers=5,
        )

        bert = AutoModel.from_pretrained("vinai/phobert-base", local_files_only=True)
        self.backbone.embeddings.word_embeddings = bert.embeddings.word_embeddings
        self.backbone.embeddings.position_embeddings = (
            bert.embeddings.position_embeddings
        )
        self.backbone.embeddings.position_ids = bert.embeddings.position_ids

        # HYPER PARAMS
        self.config = self.backbone.config
        self.segment_length = bert.config.max_position_embeddings
        self.max_length = max_length
        self.overlap = overlap

        self.project = nn.Linear(bert.config.max_position_embeddings, max_length)
        n_slices = len(
            partition_slice(self.max_length, self.segment_length, self.overlap)
        )
        self.coef = nn.Parameter(torch.rand(1, n_slices))

        # DOWNSTREAM
        self.dropout = nn.Dropout(0.1)
        self.n_classes = len(fields)

        self.rel_s = RelationTagger(
            hidden_size=self.backbone.config.hidden_size,
            n_fields=self.n_classes,
        )

        self.rel_g = RelationTagger(
            hidden_size=self.backbone.config.hidden_size,
            n_fields=self.n_classes,
        )

    def forward(self, batch):
        batch = BatchEncoding(batch)
        slices = partition_slice(self.max_length, self.segment_length, self.overlap)
        # Calculate full size hidden state
        hidden_states = torch.cat(
            [
                self.backbone(
                    input_ids=batch.input_ids[:, i],
                    bbox=batch.bbox[:, i],
                    attention_mask=batch.attention_mask[:, i],
                ).last_hidden_state.unsqueeze(1)
                for i in slices
            ],
            dim=1,
        )
        weights = torch.repeat_interleave(self.coef, batch.input_ids.size(0), dim=0)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        hidden_state = torch.sum(hidden_states * weights, dim=1)
        latent = self.project(hidden_state.transpose(-1, -2)).transpose(-1, -2)
        latent = self.dropout(latent)

        rel_s = self.rel_s(self.dropout(latent))
        rel_g = self.rel_g(self.dropout(latent))

        if "labels" in batch:
            input_masks = batch.are_box_first_tokens < 2
            labels_s = batch.labels[:, 0].contiguous()
            labels_g = batch.labels[:, 1].contiguous()
            loss_s = self.spade_loss(rel_s, labels_s, input_masks)
            loss_g = self.spade_loss(rel_g, labels_g, input_masks)
        else:
            loss_s = loss_g = None

        # true_rel_g = torch.softmax(rel_g, dim=1)[:, 0:1]
        return Namespace(
            rel=[rel_s, rel_g],
            loss=[loss_s, loss_g],
        )

    def spade_loss(self, rel, labels, input_masks):
        bsz, n, i, j = rel.shape
        bsz, i1, j1 = labels.shape
        assert i == i1
        assert j == j1
        lf = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1], device=rel.device))
        # lf = nn.CrossEntropyLoss()
        true_lengths = true_length(input_masks)
        loss = 0
        labels = labels.type(torch.long)
        for b in range(bsz):
            nc = true_lengths[b]
            nr = nc + self.n_classes
            loss += lf(rel[b : b + 1, :, :nr, :nc], labels[b : b + 1, :nr, :nc])
            # loss += lf(rel[b : b + 1], labels[b : b + 1])
        return loss


class SpadeDataset(Dataset):
    def __init__(self, tokenizer, config, jsonl):
        super().__init__()
        with open(jsonl) as f:
            data = [json.loads(line) for line in f.readlines()]

        self.raw = data
        self.fields = data[0]["fields"]
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