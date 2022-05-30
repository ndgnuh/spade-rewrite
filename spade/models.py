import torch
import numpy as np
from torch import nn
from dataclasses import dataclass
from typing import Optional, List, Dict
from argparse import Namespace
from transformers import AutoTokenizer as AutoTokenizer_, BatchEncoding, AutoModel
from .graph_utils import expand_relation
from .bros.bros import BrosModel
from .box import Box
try:
    from functools import cache
except Exception:
    from functools import lru_cache as cache


@dataclass
class SpadeData:
    """Unified model input data

    This struct aims to provide an unified input to the preprocessing
    process. Since difference OCR engine has difference outputs and
    we have yet decided on one, we can just translate each of the OCR
    outputs to this struct and write the preprocessing functionality once.

    Attributes
    ----------
    texts: List[str]
        Texts inside each bounding boxes
    boxes: List[Box]
        List of bounding box in unifed format (see spade.box.Box)
    width: int
        Image width
    height: int
        Image height
    relations: Optional[List]
        List of relation matrices, there could be many relations.
        This is equivalent to the label (which the model have to learn).
        Inference data might not have label.
    """
    texts: List[str]
    boxes: List[Box]
    width: int
    height: int
    relations: Optional[List]


class RelationTagger(nn.Module):
    def __init__(self, n_fields, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_size, hidden_size)
        self.tail = nn.Linear(hidden_size, hidden_size)
        self.field_embeddings = nn.Parameter(
            torch.rand(1, n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, enc):

        enc_head = self.head(enc)
        enc_tail = self.tail(enc)

        batch_size = enc_tail.size(0)
        field_embeddings = self.field_embeddings.expand(batch_size, -1, -1)
        enc_head = torch.cat([field_embeddings, enc_head], dim=1)

        score_0 = torch.matmul(
            enc_head, self.W_label_0(enc_tail).transpose(1, 2))
        score_1 = torch.matmul(
            enc_head, self.W_label_1(enc_tail).transpose(1, 2))

        return torch.cat([score_0.unsqueeze(1), score_1.unsqueeze(1)], dim=1)


class SpadeLoss(nn.Module):
    """Spade Loss function

    it's basically cross entropy but weighted, the default weight is
    [0.1, 1], which bias the "has edge" channel.

    Parameters
    ----------
    relations: torch.Tensor
        logits of shape batch * n_rel * (fields + seq) * seq
    labels: torch.Tensor
        ground truth relation of shape batch * (fields + seq) * seq
    input_masks: Optinal[torch.Tensor]
        the mask for true input (input wihtout padding), if this is None
        the loss will also includes padding loss.
    """

    def __init__(self, num_fields, weight=[0.1, 1]):
        super().__init__()
        weight = torch.tensor(weight)
        self.loss = nn.CrossEntropyLoss(weight=weight)
        self.num_fields = num_fields

    def true_length(self, mask):
        batch = len(mask.shape) == 2
        if batch:
            return [self.true_length(m) for m in mask]
        return torch.where(mask == 1)[-1][-1]

    def forward(self, relations, labels, input_masks=None):
        bsz, n, i, j = relations.shape
        bsz, i1, j1 = labels.shape
        assert i == i1
        assert j == j1
        # lf = nn.CrossEntropyLoss()
        if input_masks is not None:
            true_lengths = self.true_length(input_masks)
        else:
            true_lengths = [j for _ in range(bsz)]
        loss = 0
        labels = labels.type(torch.long)
        for b in range(bsz):
            nc = true_lengths[b]
            nr = nc + self.num_fields
            loss += self.loss(relations[b: b + 1, :, :nr, :nc],
                              labels[b: b + 1, :nr, :nc])
        return loss


@cache
def AutoTokenizer(*args, **kwargs):
    """Return a pretrained tokenizer

    This function is the cached version of AutoTokenizer counter part.
    """
    return AutoTokenizer_.from_pretrained(*args, **kwargs)


def map_token(tokenizer, texts: List, offset: int = 0) -> List[List[int]]:
    """Map word index to token index

    Parameters
    ------------
    tokenizer:
        transformers tokenizer
    texts: List[str]
        list of texts
    offset: int = 0
        label offset
    """
    map = [[i] for i in range(offset)] + [[] for _ in texts]

    current_node_idx = offset
    for (i, text) in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            map[i + offset].append(current_node_idx)
            current_node_idx += 1
    return map


def preprocess(config: Dict,
               data: SpadeData):
    """Preprocess data before put inside spade

    Parameters:
    ----------
    config: dict
        Configuration, must have the keys:
        - tokenizer: str (tokenizer name)
        - max_position_embeddings: int
        - bbox_type: Box.Type
        - relation_expansions: (unused for now)
    data: SpadeData
        Data to be processed
    """
    # Context
    texts = data.texts
    boxes = data.boxes
    width = data.width
    height = data.height
    relations = data.relations
    tokenizer = AutoTokenizer(config['tokenizer'])

    boxes = [b.normalize(width, height) for b in boxes]
    # token_map = G.map_token(tokenizer, texts, offset=len(config['num_fields']))

    #== TOKENIZE ==#
    tokens = []
    token_boxes = []
    token_types = []
    for text, box in zip(texts, boxes):
        text_tokens = tokenizer.tokenize(text)
        num_tokens = len(text_tokens)
        tokens.extend(text_tokens)
        token_boxes.extend([box] * num_tokens)
        token_types.extend([1] + [0] * (num_tokens - 1))
    input_masks = [1] * len(tokens)

    if relations is not None:
        # PROBLEM:
        # Each relation has their own way of expanding the graph
        # so we define some extra input here
        expansions = config['relation_expansions']
        for i, relation in enumerate(relations):
            relations[i] = \
                expand_relation(relation, token_map, **expansions[i])

    # PADDING
    special_tokens_count = 2
    true_length = config['max_position_embeddings'] - special_tokens_count
    if len(tokens) > true_length:
        tokens = tokens[:true_length]
        token_boxes = token_boxes[:true_length]
        token_types = token_types[:true_length]
        input_masks = input_masks[:true_length]

    # SEP TOKEN
    tokens += [tokenizer.sep_token]
    input_masks += [1]
    token_boxes += [Box.sep_token(width, height)]
    token_types += [1]

    # CLS token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [Box.cls_token()] + token_boxes
    input_masks = input_masks + [0]
    token_types = [2] + token_types

    # PAD TOKEN
    padding_length = config['max_position_embeddings'] - len(tokens)
    tokens += [tokenizer.pad_token] * padding_length
    input_masks += [0] * padding_length
    token_boxes += [Box.pad_token()] * padding_length
    token_types += [3] * padding_length

    # BOUNDING BOX FORMAT
    token_boxes = [getattr(b, config['bbox_type']) for b in token_boxes]

    # INPUT IDS
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    def tensorize(x):
        return torch.tensor((x))

    return BatchEncoding({
        "input_ids": tensorize(input_ids).unsqueeze(0),
        "attention_mask": tensorize(input_masks).unsqueeze(0),
        "token_type_ids": tensorize(token_types).unsqueeze(0),
        "bbox": tensorize(token_boxes).unsqueeze(0),
    })


class BrosSpade(nn.Module):
    """Model include clovaai/BROS backbone + clovaai/SPADE head.
    """

    def __init__(self, config, fields, **kwargs):
        """
        WARNING: this doc is written as-is, the code doesn't follow this one (yet)
        Paramters
        ----------
        config: Dict
            the model configuration, see extra configuration keys below,
            other configuration will derive from model config on huggingface

        Configuration keys
        ----------
        num_fields: int
            Number of classes
        backbone: str
            Backbone name (for language hybrid layers)
        bros_backbone: str
            BROS backbone name (default: "naver-clova-ocr/bros-base-uncased")
        local_files_only: bool
            Whether to query hugging face server (default: False)
        hidden_dropout_prob: float
            Hidden dropout probability (default: 0.1)
        """
        super().__init__()
        bert = config._name_or_path
        self.config = config
        self.num_fields = num_fields = len(fields)
        self.backbone = BrosModel.from_pretrained(
            "naver-clova-ocr/bros-base-uncased")
        bert = AutoModel.from_pretrained(bert)
        self.backbone.embeddings.word_embeddings = bert.embeddings.word_embeddings
        self.dropout = nn.Dropout(0.1)

        self.rel_s = RelationTagger(
            hidden_size=config.hidden_size,
            n_fields=self.num_fields,
        )

        self.rel_g = RelationTagger(
            hidden_size=config.hidden_size,
            n_fields=self.num_fields,
        )
        self.loss = SpadeLoss(num_fields=num_fields)

    def forward(self, batch):
        batch = BatchEncoding(batch)
        outputs = self.backbone(
            input_ids=batch.input_ids,
            bbox=batch.bbox,
            attention_mask=batch.attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        rel_s = self.rel_s(self.dropout(last_hidden_state))
        rel_g = self.rel_g(self.dropout(last_hidden_state))

        if 'labels' in batch:
            input_masks = batch.token_types_ids < 2
            labels_s = batch.labels[:, 0].contiguous()
            labels_g = batch.labels[:, 1].contiguous()
            loss_s = self.loss(rel_s, labels_s, input_masks)
            loss_g = self.loss(rel_g, labels_g, input_masks)
        else:
            loss_s = loss_g = None

        # true_rel_g = torch.softmax(rel_g, dim=1)[:, 0:1]
        return Namespace(
            logits=[rel_s, rel_g],
            relations=[rel_s.argmax(dim=1), rel_g.argmax(dim=1)],
            loss=[loss_s, loss_g],
        )


@cache
def group_name(field):
    """Return group name

    Example: "menu.name" -> "menu", "phone" -> "phone"

    Paramters:
    ---------
    field: str
        Name of field
    """
    if "." in field:
        return field.split(".")[0]
    else:
        return field


def post_process(tokenizer, relations, batch, fields):
    """
    Process logits into human readable outputs

    Parameters:
    ----------
    tokenizer:
        Tokenizer
    relations: List[torch.Tensor]
        Model relation outputs
    batch: transformers.BatchEncoding
        The preprocessed inputs
    fields: List[str]
        Fields' name
    """

    # Convert to numpy because we use matrix indices in a dict
    rel_s, rel_g = relations
    rel_s = rel_s[0].cpu().numpy()
    rel_g = rel_g[0].cpu().numpy()

    nfields = len(fields)
    input_ids = batch.input_ids[0].cpu().tolist()
    input_masks = batch.token_type_ids < 2
    input_masks = input_masks[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    nodes = range(len(fields + input_ids))

    itc_rel = rel_s[:nfields, :]
    itc = {}
    for (i, j) in zip(*np.where(itc_rel)):
        itc[j] = i

    # subsequence token classification
    visited = {}
    tails = {i: [] for i in itc}

    # tail nodes
    has_loop = False

    def visit(i, head, depth=0):
        if visited.get(i, False):
            return
        visited[i] = True
        tails[head].append(i)
        for j in np.where(rel_s[i + nfields, :])[0]:
            visit(j, head=head, depth=depth + 1)

    for i in itc:
        visit(i, i)

    # groups head nodes
    groups = {}
    has_group = {i: False for i in itc}
    for (i, j) in zip(*np.where(rel_g[nfields:, :])):
        # print(i, j, tokens[i], tokens[j])
        if i not in groups:
            groups[i] = [i]
            has_group[i] = True
        groups[i].append(j)
        has_group[j] = True

    # each group contain the head nodes
    groups = list(groups.values())

    # Standalone label which doesn't have group
    standalones = {}
    for (i, field_id) in itc.items():
        if has_group[i]:
            continue
        g = group_name(fields[field_id])
        standalone = standalones.get(g, [])
        standalone.append(i)
        standalones[g] = standalone
    standalones = [l for l in standalones.values() if len(l) > 0]
    groups = standalones + groups

    # Tokenizer and add label to each group
    classification = []
    ignore_input_ids = [
        tokenizer.pad_token_id,
        tokenizer.sep_token_id,
        tokenizer.cls_token_id,
    ]
    for head_nodes in groups:
        # Get the tails tokens and concat them to full text
        current_classification = []
        for i in head_nodes:
            if i not in tails:
                # predicted Rel G may not connect to a head node
                continue
            tail = tails[i]
            input_ids_i = [input_ids[j] for j in tail]
            input_ids_i = [
                id for id in input_ids_i if id not in ignore_input_ids]
            texts = tokenizer.decode(input_ids_i)
            field = fields[itc[i]]
            # ignore empty fields
            if len(texts) > 0:
                # change dict to tuple for better data structure
                # or change current_classification to dict
                current_classification.append({field: texts})

        if len(current_classification) > 0:
            classification.append(current_classification)

    return classification
