import os
import importlib
import yaml
import numpy as np
import torch

# CONFIG RELATED


class AttrDict(dict):
    """
    A dictionary that allow access by attribute.
    This basically has the same semantic of a Lua table.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for (k, v) in self.items():
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)


def read_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    return AttrDict(config)


def load_code_path(path):
    module_path, name = os.path.splitext(path)
    name = name[1:]  # Remove '.' in the extension
    module = importlib.import_module(module_path)
    return getattr(module, name)


# SCORE RELATED

def ensure_numpy(x):
    # Convert to numpy, or just stay as numpy
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def ensure_nz(x):
    # Ensure x is not zero
    return x + 1e-6 if x == 0 else x


def scores(label, pr):
    """
    Return metrics between binary prediction and label
    - accurary
    - precision
    - recall
    - f1
    """
    pr = ensure_numpy(pr)
    label = ensure_numpy(label)
    tp = ((pr == 1) * (label == 1)).sum()
    tn = ((pr == 0) * (label == 0)).sum()
    fp = ((pr == 1) * (label == 0)).sum()
    fn = ((pr == 0) * (label == 1)).sum()
    return AttrDict(
        accuracy=(tp + tn) / ensure_nz(tp + tn + fp + fn),
        precision=tp / ensure_nz(tp + fp),
        recall=tp / ensure_nz(tp + fn),
        f1=2 * tp / ensure_nz(2 * tp + fp + fn),
    )
