from transformers.configuration_utils import PretrainedConfig
from torch_geometric import nn as gnn
from torch import nn
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

    def forward(self, enc, edge_index, edge_weights=None):
        _, n_edge = edge_index.shape
        enc_head = self.head(enc, edge_index)
        enc_tail = self.tail(enc, edge_index)

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


class ProtoGraphConfig(PretrainedConfig):

    def __init__(self,
                 n_layers=5,
                 d_embed=768,
                 d_model=768,
                 n_head=16,
                 vocab_size=64001,
                 n_labels=10,
                 p_dropout=0.1,
                 n_relation=2,
                 u_text=20,
                 u_dist=120):
        super().__init__(n_layers=n_layers,
                         d_model=d_model,
                         d_embed=d_embed,
                         n_head=n_head,
                         vocab_size=vocab_size,
                         n_labels=n_labels,
                         p_dropout=p_dropout,
                         n_relation=n_relation,
                         u_text=u_text,
                         u_dist=u_dist)


class ProtoGraphLayer(nn.Module):

    def __init__(self, d_model, n_head, num_layer, update_link):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.update_link = update_link
        # self.gnn = gnn.GCN(
        #     d_model,
        #     d_model,
        #     num_layers=3,
        #     dropout=0.1,
        # )
        self.gnns = nn.ModuleList()
        for i in range(num_layer):
            if i == num_layer - 1:
                layer = gnn.GCNConv(d_model,
                                    d_model,
                                    improved=True,
                                    add_self_loops=False)
            else:
                layer = gnn.GATConv(d_model, d_model, n_head, bias=False)
            self.gnns.append(layer)

        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)
        if self.update_link:
            self.link_predict = LinkPredict(d_model)

    def forward(self, hidden, edge_index, edge_weights=None):
        if hidden.dim() == 3:
            hidden = hidden[0, :, :]
        if edge_index.dim() == 3:
            edge_index = edge_index[0, :, :]
        if edge_weights is not None and edge_weights.dim() == 2:
            edge_weights = edge_weights[0, :]
        hidden = self.layernorm(hidden)
        n_node = hidden.shape[0]
        for layer in self.gnns:
            if isinstance(layer, gnn.GCNConv):
                hidden = layer(hidden, edge_index, edge_weights)
            else:
                hidden = layer(hidden, edge_index)
                hidden = hidden.reshape(n_node, self.d_model, self.n_head)
                hidden = hidden.mean(dim=-1)
        if self.update_link:
            edge_index, edge_weights = self.link_predict(
                hidden, edge_index, edge_weights)
        return hidden, edge_index, edge_weights


class ProtoGraphModel(nn.Module):

    def __init__(self, config, embeddings=None):
        super().__init__()
        self.embeddings = embeddings
        self.config = config
        self.layers = nn.ModuleList()

        n_gcn_layers = [3] * 6
        update_links = [False, False, True] * 6
        for n, update_link in zip(n_gcn_layers, update_links):
            layer = ProtoGraphLayer(config['d_model'], config['n_head'], n,
                                    update_link)
            self.layers.append(layer)

        self.emb_proj = nn.Linear(config['d_embed'], config['d_model'])

    def forward(self, inputs, edge_index, edge_weights=None):
        if self.embeddings is not None:
            hidden = self.embeddings(inputs)
        else:
            hidden = inputs

        hidden = self.emb_proj(hidden)

        for layer in self.layers:
            new_hidden, edge_index, edge_weights = layer(
                hidden, edge_index, edge_weights)
            hidden = new_hidden + hidden

        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        return hidden, edge_index, edge_weights
