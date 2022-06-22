import numpy as np
import networkx as nx
from torch import nn
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional
import spade_gcn.graph_stuff as G
import torch
import json
from argparse import Namespace
from spade_gcn.gcn_1 import modeling_gcn
from spade_gcn.gcn_1.modeling_gcn import ProtoGraphConfig


def get_token_edge_index(token_map):
    edge_index = []
    for tkm in token_map:
        edge_index.extend(list(zip(tkm, tkm[1:])))
    return tensorize(edge_index).transpose(0, 1)


def get_text_features(texts, token_map):
    """
    gets text features 

    Args: texts: List[str]
    Returns: n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special
    """
    # data = df['Object'].tolist()
    special_chars = [
        '&', '@', '#', '(', ')', '-', '+', '=', '*', '%', '.', ',', '\\', '/',
        '|', ':'
    ]

    # character wise
    n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_special = [],[],[],[],[],[]

    # token level feature
    TOKEN_PIECE = '@@'
    TOKEN_TYPE_START = 0
    TOKEN_TYPE_END = 1
    TOKEN_TYPE_MID = 2
    TOKEN_TYPE_FULL = 3
    token_types = [None for text in texts]  # start, end, middle, full

    for (i, text) in enumerate(texts):
        text = text.replace("@@", "").replace("##", "")
        lower, upper, alpha, spaces, numeric, special = 0, 0, 0, 0, 0, 0
        for char in text:
            if char.islower():
                lower += 1
            # for upper letters
            if char.isupper():
                upper += 1
            # for white spaces
            if char.isspace():
                spaces += 1
            # for alphabetic chars
            if char.isalpha():
                alpha += 1
            # for numeric chars
            if char.isnumeric():
                numeric += 1
            if char in special_chars:
                special += 1

        tkm = [tkm for tkm in token_map if i in tkm]
        len_tkm = len(tkm)
        if len_tkm == 1:
            token_types[i] = TOKEN_TYPE_FULL
        else:
            index = tkm.index(i)
            if index == 0:
                token_types[i] = TOKEN_TYPE_START
            elif index == len_tkm - 1:
                token_types[i] = TOKEN_TYPE_END
            else:
                token_types[i] = TOKEN_TYPE_MID

        n_lower.append(lower)
        n_upper.append(upper)
        n_spaces.append(spaces)
        n_alpha.append(alpha)
        n_numeric.append(numeric)
        n_special.append(special)

    result = {
        'n_lower': n_lower,
        'n_upper': n_upper,
        'n_alpha': n_alpha,
        'n_spaces': n_spaces,
        'n_numeric': n_numeric,
        'n_special': n_special,
        'token_types': token_types
    }
    #features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])
    # df['n_upper'],df['n_alpha'],df['n_spaces'],\
    # df['n_numeric'],df['n_special'] = n_upper, n_alpha, n_spaces, n_numeric,n_special

    return result


def batch_consine_sim(batch):
    score = torch.einsum("bih,bjh->bij", batch, batch)
    inv_norm = 1 / torch.norm(batch, dim=-1)
    return torch.einsum("bij,bi,bj->bij", score, inv_norm, inv_norm)


def tensorize(x):
    try:
        return torch.tensor(np.array(x))
    except Exception:
        return torch.tensor(x)


def get_dist(cx, cy):
    dx = cx[None, :] - cx[:, None]
    dy = cy[None, :] - cy[:, None]
    return (dx**2 + dy**2)**(0.5)


def gen_box_graph(bboxes):
    # [[x1, y1], ... [x4, y4]]
    # Use numpy.ndarray because it has faster indexing for some reasons
    n = bboxes.shape[0]
    xmaxs = bboxes[:, :, 0].max(axis=1)
    xmins = bboxes[:, :, 0].min(axis=1)
    ymaxs = bboxes[:, :, 1].max(axis=1)
    ymins = bboxes[:, :, 1].min(axis=1)
    xcentres = bboxes[:, :, 0].mean(axis=1)
    ycentres = bboxes[:, :, 1].mean(axis=1)
    heights = ymaxs - ymins
    widths = xmaxs - xmins

    def is_top_to(i, j):
        is_top = (ycentres[j] - ycentres[i]) > ((heights[i] + heights[j]) / 3)
        is_on = abs(xcentres[i] - xcentres[j]) < ((widths[i] + widths[j]) / 3)
        result = is_top and is_on
        return result

    def is_left_to(i, j):
        is_left = (xcentres[i] - xcentres[j]) > ((widths[i] + widths[j]) / 3)
        is_to = abs(ycentres[i] - ycentres[j]) < (
            (heights[i] + heights[j]) / 3)
        return is_left and is_to


#     def is_top_to(i, j):
#         result = ymaxs[i] < ymins[j]
#         return result

#     def is_left_to(i, j):
#         return (xmaxs[i] < xmins[j])

# <L-R><T-B>

    horz = np.zeros((n, n), dtype=int)
    vert = np.zeros((n, n), dtype=int)
    eps = 0.05
    for i in range(n):
        for j in range(n):
            dist = np.sqrt((xcentres[i] - xcentres[j])**2 +
                           (ycentres[i] - ycentres[j])**2)

            # This is to fool the min spanning tree alg
            dist = dist - eps
            if is_left_to(i, j):
                horz[i, j] = dist
            if is_top_to(i, j):
                vert[i, j] = dist
    horz = nx.minimum_spanning_tree(nx.from_numpy_matrix(horz))
    vert = nx.minimum_spanning_tree(nx.from_numpy_matrix(vert))
    horz = torch.tensor(nx.to_numpy_array(horz), dtype=torch.float)
    vert = torch.tensor(nx.to_numpy_array(vert), dtype=torch.float)
    return horz, vert


def parse_input(
        width,
        height,
        words,
        actual_boxes,
        tokenizer,
        config,
        label,
        fields,
        cls_token_box=[0] * 8,  # useless
        sep_token_box=None,  # useless
        pad_token_box=[0] * 8,  # useless
):
    label_is_none = label is None
    if label is None:
        label = torch.zeros(
            2,
            1,
            1,
            dtype=torch.long,
        )
    boxes = np.array(actual_boxes)
    # print("boxes: ",boxes.shape)
    label = tensorize(label)
    token_map = modeling_gcn.map_token(tokenizer, words, offset=len(fields))
    rel_s = tensorize(label[0])
    rel_g = tensorize(label[1])
    token_rel_s = modeling_gcn.expand(rel_s,
                                      token_map,
                                      lh2ft=True,
                                      in_tail=True,
                                      in_head=True)
    token_rel_g = modeling_gcn.expand(rel_g, token_map, fh2ft=True)
    label = torch.cat(
        [token_rel_s.unsqueeze(0),
         token_rel_g.unsqueeze(0)],
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

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    total_len = len(input_ids)
    assert len(input_ids) == total_len
    assert len(segment_ids) == total_len
    assert len(token_boxes) == total_len
    assert len(token_actual_boxes) == total_len
    assert len(are_box_first_tokens) == total_len

    # token map without offset
    token_map = modeling_gcn.map_token(tokenizer, words, offset=0)

    # Tokenizer features
    token_edge_index = get_token_edge_index(token_map)

    # character level text features
    text_features = get_text_features(words, token_map)
    for k, features in text_features.items():
        aux = []
        for (i, feature) in enumerate(features):
            aux.extend([feature] * len(token_map[i]))
        text_features[k] = torch.tensor(np.array(aux))

    # Edge features
    gh, gv = gen_box_graph(boxes)
    gh = gh / width
    gv = gv / height
    gv = modeling_gcn.expand(gv, token_map, lh2ft=True)
    gh = modeling_gcn.expand(gh,
                             token_map,
                             in_head=True,
                             in_tail=True,
                             lh2ft=True)
    g = gh + gv
    edge_index_1 = torch.tensor(list(zip(*torch.where(g != 0)))).transpose(
        0, 1)
    # edge_index_2 = torch.tensor(list(zip(*torch.where(g == -1)))).transpose(
    #     0, 1)
    edge_weights_1 = torch.ones_like(edge_index_1[0])
    # edge_weights_2 = torch.ones_like(edge_index_2[0]) * 2
    # token_edge_weights = torch.ones_like(token_edge_index[0]) * 3
    edge_index = torch.cat([edge_index_1], dim=-1)
    edge_weights = torch.cat([edge_weights_1], dim=-1)
    # edge_weights = torch.tensor(
    #     [g[i, j] for (i, j) in zip(*torch.where(g != 0))])

    # Spatial features
    u_dist = config.u_dist
    rxs = [sum([p[0] for p in box]) / 4 / width for box in token_boxes]
    rys = [sum([p[1] for p in box]) / 4 / height for box in token_boxes]
    rx_ids = tensorize([int(rx * u_dist) for rx in rxs])
    ry_ids = tensorize([int(ry * u_dist) for ry in rys])

    ret = {
        "text_tokens": tokens,
        "input_ids": tensorize(input_ids),
        "token_type_ids": tensorize(segment_ids),
        "edge_index": edge_index,
        "labels": label,
        "edge_weights": edge_weights,
        "are_box_first_tokens": tensorize(are_box_first_tokens),
        "rx_ids": rx_ids,
        "ry_ids": ry_ids,
        "token_edge_index": token_edge_index,
    }
    ret.update(text_features)
    if label_is_none and 'labels' in ret:
        ret.pop('labels')

    return ret


def batch_parse_input(tokenizer, config, batch_data):
    batch = []
    text_tokens = []
    for d in batch_data:
        texts = d["text"]
        actual_boxes = d["coord"]
        label = d["label"]
        fields = d["fields"]
        features = parse_input(d["img_sz"]["width"],
                               d["img_sz"]["height"],
                               texts,
                               actual_boxes,
                               tokenizer,
                               config,
                               label=label,
                               fields=fields)
        text_tokens.append(features.pop("text_tokens"))
        batch.append(features)

    return batch, text_tokens


class RelationTagger(nn.Module):

    def __init__(self, n_fields, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = nn.Linear(hidden_size, hidden_size)
        self.tail = nn.Linear(hidden_size, hidden_size)
        self.field_embeddings = nn.Parameter(torch.rand(n_fields, hidden_size))
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, enc):
        enc_head = self.head(enc)
        enc_tail = self.tail(enc)
        enc_head = torch.cat([self.field_embeddings, enc_head], dim=0)

        score_0 = torch.matmul(enc_head,
                               self.W_label_0(enc_tail).transpose(0, 1))
        score_1 = torch.matmul(enc_head,
                               self.W_label_1(enc_tail).transpose(0, 1))

        score = torch.cat(
            [
                score_0.unsqueeze(0),
                score_1.unsqueeze(0),
            ],
            dim=0,
        )
        return score


class GCNSpade(nn.Module):

    def __init__(self, config, embeddings=None):
        super().__init__()
        self.embeddings = NodeEmbedding(config['d_embed'],
                                        config['u_text'],
                                        config['u_dist'],
                                        pretrain_we=config['tokenizer'],
                                        vocab_size=config['vocab_size'])
        self.backbone = modeling_gcn.ProtoGraphModel(config, embeddings=None)
        self.n_classes = config['n_labels']
        self.dropout = nn.Dropout(0.1)

        self.rel_s = RelationTagger(
            hidden_size=config['d_model'],
            n_fields=self.n_classes,
        )

        self.rel_g = RelationTagger(
            hidden_size=config['d_model'],
            n_fields=self.n_classes,
        )

        self.proj = nn.Linear(config['d_model'], config['d_model'])

    def forward(self, batch):
        inputs = self.embeddings(**batch)
        last_hidden_state, _, _ = self.backbone(
            inputs,
            batch['edge_index'],
            batch['edge_weights'],
        )
        last_hidden_state = self.proj(last_hidden_state)
        last_hidden_state = self.dropout(last_hidden_state)

        rel_s = self.rel_s(last_hidden_state)
        rel_g = self.rel_g(last_hidden_state)
        if 'labels' in batch:
            labels_s = batch.labels[:, 0].contiguous()
            labels_g = batch.labels[:, 1].contiguous()
            loss_s = self.spade_loss(rel_s, labels_s)
            loss_g = self.spade_loss(rel_g, labels_g)
            loss = [loss_s, loss_g]
        else:
            loss = None

        # true_rel_g = torch.softmax(rel_g, dim=1)[:, 0:1]
        return Namespace(
            prob=[rel_s, rel_g],
            relation=[rel_s.argmax(dim=0),
                      rel_g.argmax(dim=0)],
            rel=[rel_s.unsqueeze(0), rel_g.unsqueeze(0)],
            loss=loss,
        )

    def spade_loss(self, rel, labels):
        lf = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1, 1], device=rel.device))
        labels = labels.type(torch.long)
        rel = rel.unsqueeze(0)
        loss = lf(rel, labels)
        return loss


class SpadeDataset(Dataset):

    def __init__(self, tokenizer, config, jsonl , fields):
        super().__init__()
        with open(jsonl, encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        for i in range(len(data)):
            data[i]["fields"]=fields
        self.raw = data
        self.fields = data[0]["fields"]
        self.nfields = len(self.fields)
        self._cached_length = len(data)
        self.features, self.text_tokens = batch_parse_input(
            tokenizer, config, data)

    def __len__(self):
        return self._cached_length

    def __getitem__(self, idx):
        features = self.features[idx]
        if isinstance(features, list):
            features = features[0]

        for (k, v) in features.items():
            if v.dim() == 3:
                features[k] = v.squeeze(0)

        x = BatchEncoding(features)
        y = features.get('labels', None)
        return x, y


class NodeEmbedding(nn.Module):

    def __init__(self,
                 d_embed,
                 u_text,
                 u_dist,
                 pretrain_we=None,
                 vocab_size=None):
        super().__init__()
        # Meta
        self.u_text = u_text - 1
        self.u_dist = u_dist - 1
        n_features = 10
        d_epart = d_embed // n_features
        d_epart_odd = d_embed - d_epart * (n_features - 1)

        # word embeddings
        if pretrain_we is not None:
            self.we = AutoModel.from_pretrained(
                pretrain_we).embeddings.word_embeddings
            self.we_proj = nn.Linear(768, d_epart_odd)
        else:
            self.we = nn.Embedding(vocab_size, d_epart)
            self.we_proj = nn.Identity()

        # Text features
        self.n_lower = nn.Embedding(u_text, d_epart)
        self.n_upper = nn.Embedding(u_text, d_epart)
        self.n_alpha = nn.Embedding(u_text, d_epart)
        self.n_spaces = nn.Embedding(u_text, d_epart)
        self.n_numeric = nn.Embedding(u_text, d_epart)
        self.n_special = nn.Embedding(u_text, d_epart)
        self.token_types = nn.Embedding(4, d_epart)

        # Position features
        self.rx = nn.Embedding(u_dist, d_epart)
        self.ry = nn.Embedding(u_dist, d_epart)

    def forward(self, input_ids, token_types, n_lower, n_upper, n_alpha,
                n_spaces, n_numeric, n_special, rx_ids, ry_ids, **kwargs):

        # Text features
        we = self.we(input_ids)
        we = self.we_proj(we)
        # print('n_lower', n_lower)
        n_lower = self.n_lower(torch.clamp(n_lower, 0, self.u_text))
        # print('n_upper', n_upper)
        n_upper = self.n_upper(torch.clamp(n_upper, 0, self.u_text))
        # print('n_alpha', n_alpha)
        n_alpha = self.n_alpha(torch.clamp(n_alpha, 0, self.u_text))
        # print('n_spaces', n_spaces)
        n_spaces = self.n_spaces(torch.clamp(n_spaces, 0, self.u_text))
        # print('n_numeric', n_numeric)
        n_numeric = self.n_numeric(torch.clamp(n_numeric, 0, self.u_text))
        # print('n_special', n_special)
        n_special = self.n_special(torch.clamp(n_special, 0, self.u_text))

        token_types = self.token_types(token_types)

        # Position features
        rx = self.rx(torch.clamp(rx_ids, 0, self.u_dist))
        ry = self.ry(torch.clamp(ry_ids, 0, self.u_dist))

        # Concat
        embeddings = [
            we,  # Token ids features
            n_lower,
            n_upper,
            n_alpha,
            n_spaces,
            n_numeric,
            n_special,  # Char features
            token_types,
            rx,
            ry  # Spatial features
        ]
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings


class NodeEmbedding2(nn.Module):

    def __init__(self, d_embed, u_text, u_dist):
        super().__init__()
        # Meta
        self.u_text = u_text - 1
        self.u_dist = u_dist - 1

        # word embeddings
        self.we = AutoModel.from_pretrained(
            "vinai/phobert-base").embeddings.word_embeddings
        self.we_proj = nn.Linear(768, d_embed)

        # Text features
        self.n_lower = nn.Embedding(u_text, d_embed)
        self.n_upper = nn.Embedding(u_text, d_embed)
        self.n_alpha = nn.Embedding(u_text, d_embed)
        self.n_spaces = nn.Embedding(u_text, d_embed)
        self.n_numeric = nn.Embedding(u_text, d_embed)
        self.n_special = nn.Embedding(u_text, d_embed)
        self.token_types = nn.Embedding(4, d_embed)

        # Position features
        self.rx = nn.Embedding(u_dist, d_embed)
        self.ry = nn.Embedding(u_dist, d_embed)

    def forward(self, input_ids, token_types, n_lower, n_upper, n_alpha,
                n_spaces, n_numeric, n_special, rx_ids, ry_ids, **kwargs):

        # Text features
        we = self.we_proj(self.we(input_ids))
        n_lower = self.n_lower(torch.clamp(n_lower, 0, self.u_text))
        n_upper = self.n_upper(torch.clamp(n_upper, 0, self.u_text))
        n_alpha = self.n_alpha(torch.clamp(n_alpha, 0, self.u_text))
        n_spaces = self.n_spaces(torch.clamp(n_spaces, 0, self.u_text))
        n_numeric = self.n_numeric(torch.clamp(n_numeric, 0, self.u_text))
        n_special = self.n_special(torch.clamp(n_special, 0, self.u_text))

        token_types = self.token_types(token_types)

        # Position features
        rx = self.rx(torch.clamp(rx_ids, 0, self.u_dist))
        ry = self.ry(torch.clamp(ry_ids, 0, self.u_dist))

        # Concat
        embeddings = [
            we,  # Token ids features
            n_lower,
            n_upper,
            n_alpha,
            n_spaces,
            n_numeric,
            n_special,  # Char features
            token_types,
            rx,
            ry  # Spatial features
        ]
        embeddings = sum(embeddings)
        return embeddings


def spade_loss(rel, labels):
    lf = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1], device=rel.device))
    labels = labels.type(torch.long)
    rel = rel.unsqueeze(0)
    loss = lf(rel, labels)
    return loss
