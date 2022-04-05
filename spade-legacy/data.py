import torch
from itertools import product
from functools import lru_cache as cache
import numpy as np
from dataclasses import dataclass


@dataclass(unsafe_hash=True, eq=True)
class Text:
    text: str
    coord: tuple
    token: bool


def rel_vectors(tokenizer, text, coord, img_width, img_height):
    """
    Return relative feature vectors from input data.

    Parameter
    ---
    tokenizer:
        transformers tokenizer
    text:
        batch of OCR text
    coord: torch.Tensor | list
        Batch of bounding boxes in 4-corner format, should be in
        the form of (batch x number_of_boxes x 2 x 4)
    img_width:
        Image's width
    img_height:
        Image's height

    Return
    ---
    rel_x:
        Relative centres-x from one box to another
    rel_y:
        Relative centres-x from one box to another
    rel_dist:
        Relative distances from one box to another
    rel_angles:
        Relative angles from one box to another
    """
    if isinstance(coord, list):
        coord = torch.tensor(coord, dtype=torch.float32)

    tokens = [tokenizer(text_, return_tensors='pt')['input_ids']
              for text_ in text]

    # Direction vector
    # It's just (
    dirs = (coord[:, 1] + coord[:, 2]) / 2 \
        - (coord[:, 0] + coord[:, 3]) / 2
    n = len(dirs)

    # Directions normalized
    tokens = torch.cat(tokens, dim=1).squeeze().unsqueeze(1)
    dirs_norm = torch.norm(dirs, dim=1).unsqueeze(1) ** -1
    dirs_norm = dirs_norm * dirs

    # Angle between boxes
    dot = cache(torch.dot)
    rel_angles = torch.tensor(
        [dot(v1, v2) for (v1, v2) in product(dirs_norm, dirs_norm)])
    rel_angles = rel_angles.reshape(n, n)
    rel_angles = torch.arccos(rel_angles)

    # Relative centers
    centres = coord.mean(dim=1)
    rel_centres = torch.cat(
        [(v1 - v2) for (v1, v2) in product(centres, centres)])
    rel_centres = rel_centres.reshape(n, n, 2)

    # Relative distance
    rel_dist = torch.norm(rel_centres, dim=-1)

    # NORMALIZE
    unit_length = (img_width**2 + img_height**2)**2 / 120

    def normalize(x):
        return torch.clip(x / unit_length, -120, 120)
    rel_centres = normalize(rel_centres)
    rel_dist = normalize(rel_dist)
    rel_angles = torch.clip(rel_angles / (2 * torch.pi) * 60, 0, 60)
    return rel_centres[:, :, 0], rel_centres[:, :, 1], rel_dist, rel_angles


def partition_indices(n, psize, overlap=0):
    assert psize > overlap
    # SPECIAL CASE
    if n <= psize - overlap:
        return [slice(0, n)]
    indices = range(0, n - overlap, psize - overlap)
    indices = list(indices)
    if indices[-1] < n:
        indices.append(indices[-1] - overlap + psize)
    ranges = [slice(i, j + overlap) for (i, j) in zip(indices, indices[1:])]
    return ranges


def parse_input(tokenizer, data, max_length=512, max_chunk=5, overlap=256, n_dist_unit=1000, tokenize=True):
    #     print(data.keys())
    # max_flatten_length = max_length * max_chunk - (max_chunk - 1) * overlap
    texts = data['text']
    coord = data['coord']
    width = data['img_sz']['width']
    height = data['img_sz']['height']

    def normalize(x, w):
        return int(round(x * n_dist_unit / w))

    def coord_center(poly):
        x = [pt[0] for pt in poly]
        y = [pt[1] for pt in poly]
        return [
            normalize(sum(x) / len(x), width),
            normalize(sum(y) / len(y), height)]

    def chunk(seq, part_indices, max_length, pad):
        chunks = [seq[i] for i in part_indices]
        last_len = len(chunks[-1])
        chunks[-1] = chunks[-1] + [pad] * (max_length - last_len)
        return chunks

    if tokenize:
        text_tokens = []
        box_tokens = []
        # header_ids = []
        for (text, coord) in zip(texts, data['coord']):
            tk = tokenizer.tokenize(text)
            text_tokens.extend(tk)
            box_tokens.extend(len(tk) * [coord])
            # header_ids += [1 if i == 0 else 0 for i, _ in enumerate(tk)]
    else:
        text_tokens = texts
        box_tokens = coord
    rn_center_x_ids = [coord_center(c)[0] for c in box_tokens]
    rn_center_y_ids = [coord_center(c)[1] for c in box_tokens]

    # INSERT SPECIAL TOKENS
    text_tokens = [tokenizer.cls_token] + text_tokens + [tokenizer.sep_token]
    rn_center_x_ids = [0] + rn_center_x_ids + [n_dist_unit]
    rn_center_y_ids = [0] + rn_center_y_ids + [n_dist_unit]
    # header_ids = [1] + header_ids + [1]
    attention_mask = [1 for _ in text_tokens]  # TODO
    original_len = len(text_tokens)
    text_tokens_ids = tokenizer.convert_tokens_to_ids(text_tokens)


#     print(len(text_tokens), len(rn_center_x_ids))
    assert len(text_tokens) == len(rn_center_x_ids)
    assert len(text_tokens) == len(rn_center_y_ids)

    # CHUNKING
    part_indices = partition_indices(len(text_tokens), max_length, overlap)
    part_indices[-1] = slice(part_indices[-1].start,
                             max(part_indices[-1].stop, max_length))
    text_tokens_ids = chunk(text_tokens_ids, part_indices,
                            max_length, tokenizer.pad_token_id)
    rn_center_x_ids = chunk(
        rn_center_x_ids, part_indices, max_length, n_dist_unit)
    rn_center_y_ids = chunk(
        rn_center_y_ids, part_indices, max_length, n_dist_unit)

    if "label" in data:
        label = data["label"]
        new_label = [
            expand_rel_s(score=label[0],
                         tokenizer=tokenizer,
                         coords=data['coord'],
                         texts=texts,
                         labels=data['fields']),
            expand_rel_g(score=label[1],
                         tokenizer=tokenizer,
                         coords=data['coord'],
                         texts=texts,
                         labels=data['fields'])
        ]
    else:
        new_label = None

    return {
        'input_ids': torch.tensor(text_tokens_ids),
        'position_ids': (torch.tensor(rn_center_x_ids), torch.tensor(rn_center_y_ids)),
        "part_indices": part_indices,
        "original_length": original_len,
        "text_tokens": text_tokens,
        "label": new_label
    }


def tuplize(list_):
    if not isinstance(list_, list):
        return list_
    else:
        return tuple([tuplize(elem) for elem in list_])


def expand_rel_s(score, tokenizer, texts, coords, labels):
    coords = tuplize(coords)
    text_nodes = [Text(text, coord, False)
                  for (text, coord) in zip(texts, coords)]
    nodes = labels + text_nodes
    nlabel = len(labels)

    # Switch graph reprensentation
    edges = []
    for (i, j) in zip(*np.where(score)):
        edges.append((nodes[i], text_nodes[j]))

    # Map text -> token
    token_map = {}
    for node in text_nodes:
        token_map[node] = [
            Text(txt, node.coord, True)
            for txt in tokenizer.tokenize(node.text)]

    def is_field_or_token(node):
        if isinstance(node, str):
            return True
        else:
            return node.token

    # Replace nodes
    for (node, token_nodes) in token_map.items():
        idx = nodes.index(node)
        for (i, token_node) in enumerate(token_nodes):
            nodes.insert(idx + i, token_node)
        nodes.remove(node)

    # Map edge -> replacement edges
    while True:
        if np.all([is_field_or_token(u) and is_field_or_token(v)
                   for (u, v) in edges]):
            break
        for (u, v) in edges:
            # IF text_node is a head node
            if u in token_map:
                edges.remove((u, v))
                tk_text_nodes = token_map[u]
                [edges.append(e)
                 for e in zip(tk_text_nodes, tk_text_nodes[1:])]
                edges.append((tk_text_nodes[-1], v))

            # IF text_node is a follow node
            elif v in token_map:
                edges.remove((u, v))
                tk_text_nodes = token_map[v]
                [edges.append(e)
                 for e in zip(tk_text_nodes, tk_text_nodes[1:])]
                edges.append((u, tk_text_nodes[0]))

    adj = np.zeros((len(nodes), len(nodes) - nlabel), int)
    for (u, v) in edges:
        i = nodes.index(u)
        j = nodes.index(v) - nlabel
        adj[i, j] = 1
    return adj


def expand_rel_g(score, tokenizer, texts, coords, labels):
    coords = tuplize(coords)
    text_nodes = [Text(text, coord, False)
                  for (text, coord) in zip(texts, coords)]
    nodes = labels + text_nodes
    nlabel = len(labels)

    # Switch graph reprensentation
    edges = []
    for (i, j) in zip(*np.where(score)):
        edges.append((nodes[i], text_nodes[j]))

    # Map text -> token
    token_map = {}
    for node in text_nodes:
        token_map[node] = [
            Text(txt, node.coord, True)
            for txt in tokenizer.tokenize(node.text)]

    def is_field_or_token(node):
        if isinstance(node, str):
            return True
        else:
            return node.token

    # Replace nodes
    for (node, token_nodes) in token_map.items():
        idx = nodes.index(node)
        for (i, token_node) in enumerate(token_nodes):
            nodes.insert(idx + i, token_node)
        nodes.remove(node)

    # Map edge -> replacement edges
    while True:
        if np.all([is_field_or_token(u) and is_field_or_token(v)
                   for (u, v) in edges]):
            break
        for (u, v) in edges:
            # IF text_node is a head node
            if u in token_map:
                edges.remove((u, v))
                tk_text_nodes = token_map[u]
                edges.append((tk_text_nodes[0], v))

            # IF text_node is a follow node
            elif v in token_map:
                edges.remove((u, v))
                tk_text_nodes = token_map[v]
                edges.append((u, tk_text_nodes[0]))

    adj = np.zeros((len(nodes), len(nodes) - nlabel), int)
    for (u, v) in edges:
        i = nodes.index(u)
        j = nodes.index(v) - nlabel
        adj[i, j] = 1

    return adj
