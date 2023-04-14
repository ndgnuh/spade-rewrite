import json
import numpy as np


def read_jsonl(file):
    with open(file) as f:
        raw_data = [json.loads(line) for line in f.readlines()]


    data = []

    for sample in raw_data:
        texts = sample['text']
        polygons = sample['coord']
        edge_index = []

        # Flatten classification label
        n_fields = len(sample['fields'])
        n_texts = len(texts)
        classes_ = np.array(sample['label'])[0, :n_fields]
        classes = [-100] * n_texts
        for idx in range(n_texts):
            if np.all(classes_[:, idx] == 0):
                classes[idx] = n_fields
            else:
                classes[idx] = np.argmax(classes_[:, idx])

        # Flatten relation
        relation = np.array(sample['label'])[0, n_fields:]
        for i, j in zip(*np.where(relation)):
            edge_index.append((i, j))
        edge_index = np.array(edge_index)

        sample = dict(
            texts = texts,
            polygons = polygons,
            classes = classes,
            edge_index = edge_index
        )
        data.append(sample)
    return data
