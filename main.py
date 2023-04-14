# from kie.models import build_model
from kie.utils import process_input, read_jsonl, Timer
from transformers import AutoTokenizer
import time
import icecream
import numpy as np
icecream.install()


sample = read_jsonl("data/sample.jsonl")[0]
ic(read_jsonl("data/sample.jsonl")[0].keys())

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
n = 1000
with Timer("Total"):
    polygons = np.array(sample['coord']) * 1.0
    polygons[..., 0] = polygons[..., 0] / sample['img_sz']['width']
    polygons[..., 1] = polygons[..., 1] / sample['img_sz']['height']
    inputs = process_input(
        tokenizer=tokenizer,
        texts=sample['text'],
        polygons=polygons
    )
# config = dict(
#     num_classes=7,
#     head_size=256
# )
# model = build_model(
#     config=config,
#     replace_word_embeddings="vinai/phobert-base"
# )

# print(model)
