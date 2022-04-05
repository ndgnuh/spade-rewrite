import yaml


class AttrDict(dict):
    """
    A dictionary that allow access by attribute.
    This basically has the same semantic of a Lua table.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for (k, v) in self.items():
            if isinstance(v, dict) and not isinstance(v, AttrDict):
                self[k] = AttrDict(v)


def read_config(config_path):
    config = None
    with open(config_path) as f:
        config = yaml.load_all(f, loader=yaml.FullLoader)

    return AttrDict(config)
