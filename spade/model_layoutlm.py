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


def batch_consine_sim(batch):
    score = torch.einsum("bih,bjh->bij", batch, batch)
    inv_norm = 1 / torch.norm(batch, dim=-1)
    return torch.einsum("bij,bi,bj->bij", score, inv_norm, inv_norm)


def tensorize(x):
    return torch.as_tensor(np.array(x))


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


class PlainRelationTagger(nn.Module):
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


class RelationTagger(nn.Module):
    def __init__(self, hidden_size, n_fields, max_position_embeddings, n_channels=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_fields = n_fields
        self.max_position_embeddings = max_position_embeddings
        self.threshold = nn.Parameter(torch.rand(1))

        self.register_buffer(
            "label_idx", torch.arange(0, self.n_fields, dtype=torch.long)
        )

        self.label_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    n_fields,
                    hidden_size,
                )
                for _ in range(self.n_channels)
            ]
        )
        self.key = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False)
                for _ in range(self.n_channels)
            ]
        )
        self.query = nn.ModuleList(
            [
                (nn.Linear(hidden_size, hidden_size, bias=False))
                for _ in range(self.n_channels)
            ]
        )
        self.value = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False)
                for _ in range(self.n_channels)
            ]
        )

    def forward(self, hidden):
        # Input: batch * seq * hidden
        bsize = hidden.shape[0]
        label_embeddings = torch.cat(
            [embed(self.label_idx).unsqueeze(0) for embed in self.label_embeddings],
            dim=0,
        )
        label_embeddings = torch.repeat_interleave(
            label_embeddings.unsqueeze(1), bsize, dim=1
        )
        key = torch.cat([key(hidden).unsqueeze(0) for key in self.key], dim=0)
        query = torch.cat([query(hidden).unsqueeze(0) for query in self.query], dim=0)
        value = torch.cat([value(hidden).unsqueeze(0) for value in self.value])
        query = torch.cat([query, label_embeddings], dim=2)
        attention = torch.einsum("cbih,cbjh->bcij", query, key)
        score = torch.einsum("cbih,bcji->bcji", value, attention)
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

    # The unsqueezed dim is the batch dim for each type
    return {
        "text_tokens": tokens,
        "input_ids": tensorize(input_ids).unsqueeze(0),
        "attention_mask": tensorize(input_mask).unsqueeze(0),
        "token_type_ids": tensorize(segment_ids).unsqueeze(0),
        "bbox": tensorize(token_boxes).unsqueeze(0),
        "actual_bbox": tensorize(token_actual_boxes).unsqueeze(0),
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


@dataclass
class SpadeOutput:
    itc_outputs: torch.Tensor
    stc_outputs: torch.Tensor
    attention_mask: torch.Tensor
    loss: Optional[torch.Tensor] = None


def hybrid_layoutlm(config_layoutlm, config_bert, layoutlm, bert, **kwargs):
    bert = partially_from_pretrained(config_bert, bert, **kwargs)
    layoutlm = partially_from_pretrained(config_layoutlm, layoutlm, **kwargs)
    layoutlm.embeddings.word_embeddings = bert.embeddings.word_embeddings
    layoutlm.embeddings.position_embeddings = bert.embeddings.position_embeddings
    return layoutlm


class LayoutLMSpade(nn.Module):
    def __init__(
        self, config_layoutlm, config_bert, layoutlm, bert, n_classes, fields, **kwargs
    ):
        super().__init__()
        self.config_bert = config_bert
        self.config_layoutlm = config_layoutlm
        # layoutlm = config_layoutlm._name_or_path
        # bert = bert._name_or_path
        self.n_classes = n_classes
        # self.backbone = partially_from_pretrained(config_bert, bert)
        self.backbone = hybrid_layoutlm(
            config_layoutlm, config_bert, layoutlm, bert, **kwargs
        )

        # Map field to groups
        field_groups = [f.split(".")[0] for f in fields]
        field_groups = sorted(set(field_groups), key=field_groups.index)
        field_groups = list(field_groups)
        # field_map = torch.zeros(len(fields), dtype=torch.long)
        # for (i, field) in enumerate(fields):
        #     field_map[i] = field_groups.index(field.split(".")[0])
        # self.field_map = field_map

        # (1) Initial token classification
        hidden_dropout_prob = config_bert.hidden_dropout_prob
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.simple = nn.Linear(config_bert.hidden_size, config_bert.hidden_size)

        # (2) Subsequent token classification
        self.rel_s = PlainRelationTagger(
            hidden_size=config_bert.hidden_size,
            n_fields=self.n_classes,
        )

        self.rel_g = PlainRelationTagger(
            hidden_size=config_bert.hidden_size,
            n_fields=self.n_classes,
        )
        # self.loss_func = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)
        self.loss_func = nn.CrossEntropyLoss()
        self.fill_value = 0
        self.sm2d = nn.Softmax2d()

        self.dropout = nn.Dropout(0.1)

    def forward(self, batch):
        batch = BatchEncoding(batch)
        outputs = self.backbone(
            input_ids=batch.input_ids,
            bbox=batch.bbox,
            attention_mask=batch.attention_mask,
            # token_type_ids=batch.are_box_first_tokens,
        )
        last_hidden_state = outputs.last_hidden_state
        # mask = batch.input_ids == 1
        # mask = torch.einsum("bi,bj->bij", mask, mask)
        # rel_g = 1 - batch_consine_sim(middle)
        rel_s = self.rel_s(self.dropout(last_hidden_state))
        rel_g = self.rel_g(self.dropout(last_hidden_state))
        # rel_g = rel_g * mask
        # rel_g = rel_g * torch.cos(rel_g)
        # rel_g = torch.sigmoid(rel_g)
        # rel_g = -torch.threshold(-rel_g, -1, -1)
        # rel_g = self.simple(last_hidden_state)
        # rel_s = self.rel_s_conv(rel_s)
        # rel_g = self.rel_g_conv(rel_g)

        # rel = torch.cat([rel_g, rel_g], dim=1)
        # labels = batch.labels[:, 1, -n:].contiguous()
        # labels = batch.labels.contiguous()
        # print(rel_g.shape)
        input_masks = batch.are_box_first_tokens < 2
        labels_s = batch.labels[:, 0].contiguous()
        labels_g = batch.labels[:, 1].contiguous()
        loss_s = self.spade_loss(rel_s, labels_s, input_masks)
        loss_g = self.spade_loss(rel_g, labels_g, input_masks)

        # true_rel_g = torch.softmax(rel_g, dim=1)[:, 0:1]
        return Namespace(
            rel=[rel_s, rel_g],
            loss=[loss_s, loss_g],
            attention_mask=batch.attention_mask,
        )

    #         out = SpadeOutput(
    #             itc_outputs=(itc_outputs),
    #             stc_outputs=(stc_outputs),
    #             attention_mask=batch.attention_mask,
    #             loss=,
    #         )
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

    #         return out
    def _get_rel_loss_no_mask(self, rel, labels):
        # print(rel.shape, labels.shape)
        # ml = labels.shape[-1]
        # rel = rel.view(-1, ml)
        # labels = labels.view(-1, ml)
        labels = labels.type(torch.float)
        losses = self.loss_func(rel, labels)
        return losses

    def _get_rel_loss(self, rel, labels, token_types, n_classes=None):
        # Input: batch * 2 * node * node
        # Input: batch * node * node
        # token_types: batch * node
        fill_value = self.fill_value
        if n_classes is None:
            n_classes = self.n_classes

        # repeat channel on labels
        # labels = labels.unsqueeze(1)
        # labels = torch.cat([labels, 1 - labels], dim=1)
        # print(labels.shape, rel.shape)

        # Mask label part
        mask_top = torch.zeros(
            [1, 1, n_classes, self.config_bert.max_position_embeddings],
            dtype=torch.bool,
            device=rel.device,
        )

        # Mask diagonal
        self_mask = torch.eye(rel.shape[-1], dtype=torch.bool, device=rel.device)
        self_mask = self_mask.unsqueeze(0).unsqueeze(0)
        self_mask = torch.cat([mask_top, self_mask], dim=2)
        # print(self_mask.shape, rel.shape, labels.shape)
        rel = rel.masked_fill(self_mask, fill_value)
        labels = labels.masked_fill(self_mask, fill_value)

        # Mask special tokens
        # Orig size: batch * seq
        special_mask = token_types > 1
        special_mask = special_mask.unsqueeze(1).unsqueeze(-1)
        special_mask = special_mask * special_mask.transpose(-1, -2)
        mask_top = torch.zeros(
            [
                special_mask.shape[0],
                1,
                n_classes,
                self.config_bert.max_position_embeddings,
            ],
            dtype=torch.bool,
            device=rel.device,
        )
        special_mask = torch.cat([mask_top, special_mask], dim=2)
        rel = rel.masked_fill(special_mask, fill_value)
        labels = labels.masked_fill(special_mask, fill_value)

        # labels = torch.broadcast_to(labels, rel.shape)

        # broadcast_to(labels, rel.shape)
        # labels = torch.cat(
        #     [labels, torch.ones_like(labels, device=labels.device)],  #! line break
        #     dim=1,
        # )
        ml = rel.shape[-1]
        return self.loss_func(rel.view(-1, ml), labels.view(-1, ml))

    # def forward(self, batch):
    #     if "text_tokens" in batch:
    #         # Text tokens
    #         batch.pop("text_tokens")
    #     batch = BatchEncoding(batch)
    #     outputs = self.backbone(
    #         input_ids=batch.input_ids,
    #         # bbox=batch.bbox,
    #         attention_mask=batch.attention_mask,
    #     )  # , token_type_ids=token_type_ids)
    #     last_hidden_state = outputs.last_hidden_state
    #     # print(last_hidden_state.shape)
    #     # last_hidden_state = last_hidden_state.transpose(-1, -2).contiguous()
    #     itc_outputs = self.itc_layer(last_hidden_state)  # .transpose(0, 1).contiguous()
    #     itc_outputs = self.act(itc_outputs)
    #     # print(itc_outputs.shape)
    #     last_hidden_state = last_hidden_state.transpose(0, 1).contiguous()
    #     stc_outputs = self.stc_layer(last_hidden_state, last_hidden_state).squeeze(0)
    #     # stc_outputs = self.threshold(stc_outputs)
    #     # itc_outputs = self.threshold(itc_outputs)
    #     # itc_labels = batch.itc_labels
    #     # itc_labels = torch.functional.onehots(itc_labels, self.n_classes)
    #     # batch.itc_labels = self.label_morph(batch.itc_labels)
    #     out = SpadeOutput(
    #         itc_outputs=(itc_outputs),
    #         stc_outputs=(stc_outputs),
    #         attention_mask=batch.attention_mask,
    #         loss=self._get_loss(itc_outputs, stc_outputs, batch),
    #     )
    #     return out

    def _get_loss(self, itc_outputs, stc_outputs, batch):
        itc_loss = self._get_itc_loss(itc_outputs, batch)
        stc_loss = self._get_stc_loss(stc_outputs, batch)
        # print("itc_loss", itc_loss.item(), torch.norm(itc_loss))
        # print("stc_loss", stc_loss.item(), torch.norm(stc_loss))
        # loss = itc_loss + stc_loss
        # loss =

        return itc_loss, stc_loss

    def _get_itc_loss(self, itc_outputs, batch):
        itc_mask = batch.attention_mask
        inv_mask = (1 - itc_mask).bool()
        itc_outputs = itc_outputs.masked_fill(inv_mask.unsqueeze(-1), -1.0)
        itc_outputs = itc_outputs.transpose(-1, -2)
        labels = batch.itc_labels

        return self.itc_loss_func(itc_outputs, labels)

    #     def _get_itc_loss(self, itc_outputs, batch):
    #         itc_mask = batch["are_box_first_tokens"].view(-1).bool()
    #         itc_mask = torch.where(itc_mask)

    #         itc_logits = itc_outputs.view(-1, self.n_classes)
    #         itc_logits = itc_logits[itc_mask]
    #         self.field_map = self.field_map.to(itc_logits.device)
    #         itc_labels = self.field_map[batch["itc_labels"]].view(-1)
    #         itc_labels = itc_labels[itc_mask]

    #         itc_loss = self.loss_func(itc_logits, itc_labels)

    #         return itc_loss

    # def _get_stc_loss(self, stc_outputs, batch):
    #     labels = batch.stc_labels
    #     outputs = stc_outputs.transpose(0, 1)
    #     # print(outputs.shape)
    #     # mask_x, mask_y = torch.where(batch.attention_mask.bool())
    #     nrel = labels.shape[1]
    #     # losses = [
    #     return self.loss_func(outputs, labels)

    # ]
    # return sum(losses) / nrel

    #         inv_atm = (1 - batch.attention_mask)[:, None, :]

    #         labels = labels.transpose(0, 1).masked_fill(inv_atm, -10000.0)
    #         outputs = labels.masked_fill(inv_atm, -10000.0)

    #         outputs = outputs.transpose(0, 1)
    #         labels = labels.transpose(0, 1)

    # nrel = outputs.shape[1]
    # return loss

    # bsize = outputs.shape[0]
    # loss = [
    #     self.loss_func(outputs[:, i, :, :], labels[:, i, :, :])
    #     for i in range(nrel)]
    # loss = 0
    # for b in range(bsize):
    #     for i in range(nrel):
    #         loss += self.loss_func(outputs[b, i, :, :], labels[b, i, :, :])
    # return sum(loss)
    # return sum([self.loss_func(stc_outputs[:, i, :, :], batch.stc_labels[:, i, :, :])
    #             for i in range(2)])

    def _get_stc_loss(self, stc_outputs, batch):
        invalid_token_mask = 1 - batch["attention_mask"]

        bsz, max_seq_length = invalid_token_mask.shape
        device = invalid_token_mask.device

        # invalid_token_mask = torch.cat(
        #     [inv_attention_mask, torch.zeros([bsz, 1]).to(device)], axis=1
        # ).bool()
        stc_outputs.masked_fill_(invalid_token_mask[:, None, :].bool(), -1.0)

        self_token_mask = torch.eye(max_seq_length, max_seq_length).to(device).bool()
        stc_outputs.masked_fill_(self_token_mask[None, :, :].bool(), -1.0)

        stc_mask = batch["attention_mask"].view(-1).bool()
        stc_mask = torch.where(stc_mask)

        stc_logits = stc_outputs.view(-1, max_seq_length)
        stc_logits = stc_logits[stc_mask]

        stc_labels = batch["stc_labels"]
        stc_labels = stc_labels.flatten()
        stc_labels = stc_labels[stc_mask]
        # print("labels", stc_labels.shape)
        # print("logits", stc_logits.shape)
        stc_labels = torch.broadcast_to(stc_labels.unsqueeze(1), stc_logits.shape)

        # print("labels", stc_labels.shape)
        # print("logits", stc_logits.shape)
        stc_loss = self.loss_func(stc_logits, stc_labels)

        return stc_loss

    def true_adj_single(itc_out, stc_out, attention_mask):
        idx = attention_mask.diff(dim=-1).argmin(dim=-1) + 1
        true_itc_out = itc_out[:idx]
        true_stc_out = stc_out[:idx, :idx]
        return torch.cat([true_itc_out, true_stc_out], dim=-1).transpose(0, 1)


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
