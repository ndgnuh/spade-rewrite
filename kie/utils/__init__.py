import json
import numpy as np
import time

from .io import *
from .datasets import *


# spetial token for polygons
polygon_token_cls = [0] * 8
polygon_token_sep = [1] * 8



def ensure_numpy(x):
    if isinstance(x, (tuple, list)):
        return np.array(x)
    else:
        return x

class Timer:
    def __init__(self, msg=None):
        self.t = 0
        self.msg = msg

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a, **k):
        b = time.perf_counter()
        if self.msg is not None:
            print(f"[{self.msg}] Ellapsed: {b - self.t:.9f}")
        else:
            print(f"Ellapsed: {b - self.t:.9f}")

def process_input(tokenizer, texts, polygons, class_ids=None, relations=None):
    # Scale polygons
    num_boxes = len(texts)
    polygons = ensure_numpy(polygons)
    polygons = polygons.reshape(num_boxes, 8)

    # initialize
    input_ids = [tokenizer.cls_token_id]
    polygon_ids = [polygon_token_cls]
    token_mapping = [0]

    # tokenize
    count = 1
    for (text, polygon) in zip(texts, polygons):
        input_ids_ = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        num_tokens = len(input_ids_)
        input_ids.extend(input_ids)
        polygon_ids.extend([polygon] * num_tokens)
        token_mapping.extend([count] * num_tokens)
        count = count + 1
        pass

    # append end token
    input_ids.append(tokenizer.sep_token_id)
    polygon_ids.append(polygon_token_sep)
    token_mapping.append(count)

    # convert to numpy
    polygon_ids = ensure_numpy(polygon_ids)
    input_ids = ensure_numpy(input_ids)
    token_mapping = ensure_numpy(token_mapping)

    # Return
    outputs = dict(
        input_ids=polygon_ids,
        polygon_ids=polygon_ids,
        token_mappig = token_mapping
    )
    return outputs
