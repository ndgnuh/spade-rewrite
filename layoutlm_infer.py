import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BatchEncoding,
    LayoutLMForTokenClassification,
)
from dataclasses import dataclass
from typing import Optional
from spade import model_layoutlm_2 as spade
import numpy as np
from scipy.stats import mode
from pprint import pprint
from torch.utils.data import Dataset, DataLoader

LAYOUTLM = "microsoft/layoutlm-base-cased"
BERT = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(BERT, local_files_only=True)
config = AutoConfig.from_pretrained(BERT)
config = AutoConfig.from_pretrained(
    LAYOUTLM, max_position_embeddings=config.max_position_embeddings
)

dataset = spade.SpadeDataset(
    tokenizer, config, "sample_data/predict.jsonl", test_mode=True
)
labels = dataset.fields + ["other", "special"]
dataloader = DataLoader(dataset, batch_size=1)

# MODEL

model = LayoutLMForTokenClassification.from_pretrained(
    LAYOUTLM, num_labels=len(labels) - 1
)
bert = AutoModel.from_pretrained(BERT)
model.layoutlm.embeddings.word_embeddings = bert.embeddings.word_embeddings
model.layoutlm.embeddings.position_embeddings = bert.embeddings.position_embeddings
model.train(False)

sd = torch.load("layoutlm-phobert-180data.pt")
model.load_state_dict(sd)


def post_process(tokenizer, logits, input_ids, token_types, labels):
    word_level_predictions = []  # let's turn them into word level predictions
    words = []

    current_prediction = []
    current_word = []

    token_predictions = logits.argmax(-1).tolist()
    for id, token_pred, next_head in zip(input_ids, token_predictions, token_types[1:]):
        if id in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]:
            continue
        current_word.append(id)
        current_prediction.append(token_pred)
        if next_head == 1:
            words.append(current_word)
            word_level_predictions.append(current_prediction)
            current_word = []
            current_prediction = []

    predict_output = []
    for (tokens, preds) in zip(words, word_level_predictions):
        final_label = labels[mode(preds).mode[0]]
        if final_label == "other":
            continue
        word = tokenizer.decode(tokens)
        # predict_output.append(f"- {word}: {final_label}")
        predict_output.append({word: final_label})
    return predict_output


def infer_batch(model, tokenizer, batch, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    with torch.no_grad():
        # out = model.to("cpu")(batch.to("cpu"))
        batch = batch.to(device)
        model = model.to(device)
        out = model(
            input_ids=batch.input_ids,
            bbox=batch.bbox,
            attention_mask=batch.attention_mask,
        )
    logits = out.logits
    print(logits.shape)
    bsize = logits.shape[0]
    # token_predictions = out.logits.argmax(-1).tolist()
    final_outputs = []
    for b in range(bsize):
        # logits_i = logits[b]
        final_output = post_process(
            tokenizer=tokenizer,
            logits=out.logits[b],
            # token_predictions=token_predictions[b],
            labels=labels,
            input_ids=batch.input_ids[b].tolist(),
            token_types=batch.are_box_first_tokens[b].tolist(),
        )
        final_outputs.append(final_output)
    return final_outputs


def main():
    raw_data = dataset.raw
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device is", device)
    for (idx, batch) in enumerate(dataloader):
        # only true because batchsize = 1
        print(f"----- image: {raw_data[idx]['data_id']} -----")
        outputs = infer_batch(model, tokenizer, BatchEncoding(batch), labels)
        pprint(outputs)


if __name__ == "__main__":
    main()
