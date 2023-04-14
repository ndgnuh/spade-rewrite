import numpy as np
from torch.utils.data import Dataset, DataLoader

from .io import read_jsonl


def ensure_numpy(x):
    if isinstance(x, (tuple, list)):
        return np.array(x)
    else:
        return x


# spetial token for polygons
polygon_token_cls = [0] * 8
polygon_token_sep = [1] * 8


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
    for text, polygon in zip(texts, polygons):
        input_ids_ = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        num_tokens = len(input_ids_)
        input_ids.extend(input_ids_)
        polygon_ids.extend([polygon] * num_tokens)
        token_mapping.extend([count] * num_tokens)
        count = count + 1
        pass

    # append end token
    input_ids.append(tokenizer.sep_token_id)
    polygon_ids.append(polygon_token_sep)
    token_mapping.append(count)

    # Check length
    assert len(polygon_ids) == len(input_ids)
    assert len(polygon_ids) == len(token_mapping)

    # convert to numpy
    polygon_ids = ensure_numpy(polygon_ids)
    input_ids = ensure_numpy(input_ids)
    token_mapping = ensure_numpy(token_mapping)

    # Return
    outputs = dict(
        input_ids=input_ids,
        polygon_ids=polygon_ids,
        token_mapping=token_mapping
    )
    return outputs


class MyDataset(Dataset):
    def __init__(self, tokenizer, index_file):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = read_jsonl(index_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        processed = process_input(
            self.tokenizer,
            texts=data['texts'],
            polygons=data['polygons'])
        return processed


class CollateFunc:
    pass


def build_dataloader(
    index_file: str,
    tokenizer,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False
):
    dataset = MyDataset(tokenizer, index_file)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle
    )
    return dataloader
