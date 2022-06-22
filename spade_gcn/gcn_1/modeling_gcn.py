from transformers.configuration_utils import PretrainedConfig
from spade_gcn.gcn_1.rev_gnn import GroupAddRev
from torch_geometric import nn as gnn
from torch import nn
from torch.nn import functional as F
import torch


def map_token(tokenizer, texts, offset=0):
    map = [[i] for i in range(offset)] + [[] for _ in texts]

    current_node_idx = offset
    for (i, text) in enumerate(texts):
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            map[i + offset].append(current_node_idx)
            current_node_idx += 1
    return map


def expand(
    rel,
    node_map,
    fh2ft=False,
    fh2lt=False,
    lh2ft=False,
    lh2lt=False,
    in_tail=False,
    in_head=False,
    fh2a=False,
):
    # rel-g: first head to first tail
    # rel-s: last head to first tail + in head + in tail
    m, n = rel.shape
    node_offset = m - n
    rel = torch.cat([torch.zeros(m, node_offset, dtype=rel.dtype), rel], dim=1)
    edges = [(i.item(), j.item()) for (i, j) in zip(*torch.where(rel))]
    new_edges = []
    for (u, v) in edges:
        us = node_map[u]
        vs = node_map[v]
        if fh2ft:
            i, j = us[0], vs[0]
            new_edges.append((i, j, rel[u, v]))
        if fh2lt:
            i, j = us[0], vs[-1]
            new_edges.append((i, j, rel[u, v]))
        if lh2lt:
            i, j = us[-1], vs[-1]
            new_edges.append((i, j, rel[u, v]))
        if lh2ft:
            i, j = us[-1], vs[0]
            new_edges.append((i, j, rel[u, v]))
        if in_head:
            for (i, j) in zip(us, us[1:]):
                new_edges.append((i, j, rel[u, v]))
        if in_tail:
            for (i, j) in zip(vs, vs[1:]):
                new_edges.append((i, j, rel[u, v]))
        if fh2a:
            for j in vs:
                new_edges.append((us[0], j, rel[u, v]))

    n = node_map[-1][-1] + 1
    new_adj = torch.zeros(n, n, dtype=rel.dtype)
    for (i, j, w) in new_edges:
        new_adj[i, j] = w
    return new_adj[:, node_offset:]


class LinkPredict(nn.Module):

    def __init__(self, hidden_size, head_p_dropout=0.1):
        super().__init__()
        self.head = gnn.GATConv(hidden_size, hidden_size)
        self.tail = gnn.GATConv(hidden_size, hidden_size)
        self.W_label_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_label_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(head_p_dropout)

    def forward(self, enc, edge_index, edge_weights=None):
        _, n_edge = edge_index.shape
        enc_head = self.dropout(self.head(enc, edge_index))
        enc_tail = self.dropout(self.tail(enc, edge_index))

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
        adj = score.argmax(dim=0)
        i, j = torch.where(adj != 0)
        edge_weights = score_1[i, j].flatten()
        edge_index = torch.cat([i[None, :], j[None, :]], dim=0)
        if edge_weights.shape[0] > n_edge:
            indices = torch.topk(edge_weights, n_edge).indices.flatten()
            edge_index = edge_index[:, indices]
            edge_weights = edge_weights[indices]
        edge_weights = (torch.sigmoid(edge_weights) + 1) / 2
        return edge_index, edge_weights


class ProtoGraphConfig(dict):

    def __init__(self, **kwargs):
        d_scales = [
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 1, 1],
        ]
        self_loops = [[False] * 10] * 30
        layer_type = "proto_layer"
        default = dict(tokenizer="bert-base-multilingual-cased",
                       rev_gnn_n_groups=2,
                       layer_type=layer_type,
                       update_links=[True] * 99,
                       d_scales=d_scales,
                       self_loops=self_loops,
                       lr=5e-5,
                       n_layers=5,
                       d_embed=768,
                       d_model=768,
                       n_head=16,
                       vocab_size=64001,
                       n_labels=2,
                       p_dropout=0.1,
                       n_relation=2,
                       u_text=30,
                       u_dist=120)
        default.update(kwargs)
        super().__init__(**default)
        self.__dict__ = self


class ProtoGraphLayer(nn.Module):

    def __init__(self, d_hiddens, n_head, update_link, self_loops):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_hiddens
        self.update_link = update_link
        self.gnns = nn.ModuleList()
        for i, (h1, h2) in enumerate(zip(d_hiddens, d_hiddens[1:])):
            if i == len(d_hiddens) - 1:
                sl = self_loops[i]
                layer = gnn.GCNConv(h1, h2, improved=True, add_self_loops=sl)
            else:
                layer = gnn.GATConv(h1, h2)
            self.gnns.append(layer)

        self.norm = nn.BatchNorm1d(d_hiddens[0])
        self.act = nn.LeakyReLU()
        if self.update_link:
            self.link_predict = LinkPredict(d_hiddens[-1])

        if d_hiddens[0] != d_hiddens[-1]:
            self.res_proj = nn.Linear(d_hiddens[0], d_hiddens[-1])
        else:
            self.res_proj = None

    def forward(self, hidden, edge_index, edge_weights=None):
        if hidden.dim() == 3:
            hidden = hidden[0, :, :]
        if edge_index.dim() == 3:
            edge_index = edge_index[0, :, :]
        if edge_weights is not None and edge_weights.dim() == 2:
            edge_weights = edge_weights[0, :]

        hidden = self.norm(hidden)
        hidden = self.act(hidden)
        if self.res_proj is None:
            res = hidden
        else:
            res = self.res_proj(hidden)
        for layer in self.gnns:
            if isinstance(layer, gnn.GATConv):
                hidden = layer(hidden, edge_index)
            else:
                hidden = layer(hidden, edge_index, edge_weights)

        if self.update_link:
            edge_index, edge_weights = self.link_predict(
                hidden, edge_index, edge_weights)
        return hidden + res, edge_index, edge_weights


class RevGNNLayer(nn.Module):
    # https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/examples/rev_gnn.py
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=True)
        self.conv = gnn.SAGEConv(in_channels, out_channels)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)


class ChevNet(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_rate=0.2, K=3):
        super().__init__()

        self.dropout_rate = dropout_rate

        self.conv1 = gnn.ChebConv(in_channels, 64, K=K)
        self.conv2 = gnn.ChebConv(64, 32, K=K)
        self.conv3 = gnn.ChebConv(32, 16, K=K)
        self.conv4 = gnn.ChebConv(16, out_channels, K=K)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_weight)),
                      p=self.dropout_rate,
                      training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index, edge_weight)),
                      p=self.dropout_rate,
                      training=self.training)
        x = F.dropout(F.relu(self.conv3(x, edge_index, edge_weight)),
                      p=self.dropout_rate,
                      training=self.training)
        x = self.conv4(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)


def get_d_hidden(d_model, list_scales):
    d_hiddens = []
    for scales in list_scales:
        d_hidden = [int(round(d_model) * scale) for scale in scales]
        d_hiddens.append(d_hidden)
    return d_hiddens


layer_map = dict(proto_layer=ProtoGraphLayer, rev_gnn=RevGNNLayer)


class ProtoGraphModel(nn.Module):

    def __init__(self, config, embeddings=None):
        super().__init__()
        self.embeddings = embeddings
        self.config = config
        self.layers = nn.ModuleList()
        self.layer_type = config["layer_type"]

        if self.layer_type == "proto_layer":
            n_gcn_layers = get_d_hidden(config['d_model'], config['d_scales'])
            update_links = config['update_links']
            self_loopss = config["self_loops"]
            for (hiddens, update_link,
                 self_loops) in zip(n_gcn_layers, update_links, self_loopss):
                layer = ProtoGraphLayer(hiddens,
                                        config['n_head'],
                                        update_link=update_link,
                                        self_loops=self_loops)
                self.layers.append(layer)
        elif self.layer_type == "rev_gnn":
            num_groups = config["rev_gnn_n_groups"]
            for _ in range(config["n_layers"]):
                conv = RevGNNLayer(
                    config["d_model"] // num_groups,
                    config["d_model"] // num_groups,
                )
                self.layers.append(GroupAddRev(conv, num_groups=num_groups))
            self.norm = nn.LayerNorm(config["d_model"],
                                     elementwise_affine=True)
        elif self.layer_type == "chebnet":
            d_model = config["d_model"]
            layer = ChevNet(d_model, d_model)
            self.layers.append(layer)
        else:
            raise Exception(f"Invalid layer type {self.layer_type}")

        self.emb_proj = nn.Linear(config['d_embed'], config['d_model'])

    def forward(self, inputs, edge_index, edge_weights=None):
        if self.embeddings is not None:
            hidden = self.embeddings(inputs)
        else:
            hidden = inputs

        hidden = self.emb_proj(hidden)

        if self.layer_type in ["proto_layer"]:
            for layer in self.layers:
                new_hidden, edge_index, edge_weights = layer(
                    hidden, edge_index, edge_weights)
                hidden = new_hidden
        elif self.layer_type == "rev_gnn":
            for (i, layer) in enumerate(self.layers):
                if edge_index.dim() == 3:
                    edge_index = edge_index.squeeze(0)
                # if hidden.dim() == 3:
                #     hidden = hidden.squeeze(0)
                hidden = layer(hidden, edge_index)
                hidden = self.norm(hidden)
            hidden = hidden.relu()
            # hidden = self.norm(hidden).relu()

        elif self.layer_type == "chebnet":
            for layer in self.layers:
                hidden = layer(hidden, edge_index, edge_weights)
        else:
            raise Exception(f"Invalid layer type {self.layer_type}")

        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        return hidden, edge_index, edge_weights
