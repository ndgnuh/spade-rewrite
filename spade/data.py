import torch
import torch.functional as F
from itertools import product
from functools import lru_cache as cache


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
