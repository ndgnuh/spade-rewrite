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
            new_edges.append((us[0], vs[0]))
        if fh2lt:
            new_edges.append((us[0], vs[-1]))
        if lh2lt:
            new_edges.append((us[-1], vs[-1]))
        if lh2ft:
            new_edges.append((us[-1], vs[0]))
        if in_head:
            for (i, j) in zip(us, us[1:]):
                new_edges.append((i, j))
        if in_tail:
            for (i, j) in zip(vs, vs[1:]):
                new_edges.append((i, j))
        if fh2a:
            for j in vs:
                new_edges.append((us[0], j))

    n = node_map[-1][-1] + 1
    new_adj = torch.zeros(n, n, dtype=torch.bool)
    for (i, j) in new_edges:
        new_adj[i, j] = 1
    return new_adj[:, node_offset:]


span_labels = dict(BEGIN=0, INSIDE=1, END=2, SINGLE=3, OTHER=4)
inv_span_labels = {v: k for (k, v) in span_labels.items()}
inv_span_labels[None] = 4


def graph2span(adj_s: torch.Tensor, adj_g: torch.Tensor, token_map: list):
    # Span adj
    adj = expand(
        adj_s + adj_g, token_map, lh2ft=True, in_head=True, in_tail=True, fh2ft=True
    )
    m, n = adj.shape
    offset = m - n

    label = [None for _ in range(m)]
    for i in range(n):
        if i < offset:
            continue

        has_incoming = torch.any(adj[offset:, i - offset] == 1).item()
        has_outgoing = torch.any(adj[i, :] == 1).item()
        is_head = torch.any(adj[:offset, i - offset] == 1).item()
        if is_head:

            if not has_outgoing:
                label[i] = span_labels["SINGLE"]
            else:
                label[i] = span_labels["BEGIN"]
            continue

        if has_outgoing and not has_incoming:
            label[i] = span_labels["BEGIN"]
            continue

        if has_incoming and not has_outgoing:
            label[i] = span_labels["END"]

            continue

        if has_incoming and has_outgoing:
            label[i] = span_labels["INSIDE"]
            continue

        label[i] = span_labels["OTHER"]

    return label[offset:]


def graph2span_classes(adj):
    m, n = adj.shape
    offset = m - n
    adj = adj[offset:, :]
    label = [0 for _ in range(n)]
    for i in range(n):
        has_incoming = torch.any(adj[:, i] == 1).item()
        if has_incoming:
            incoming = torch.where(adj[:, i])[0]
            assert incoming.shape[0] == 1
            label[i] = incoming[0].item()

    return label


# span = g_to_span(span_adj)
# new_adj = expand(torch.tensor(data.label[0]), map_token(tokenizer, data.text, offset=len(data.fields)), lh2ft=True, in_head=True, in_tail=True)
# show_con(new_adj, data.fields, tokens)
