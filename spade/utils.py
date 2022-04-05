from pprint import pformat
import os
import importlib
import yaml


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
