#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import traceback

# from importlib import reload
# reload(spade)
import os
import tqdm
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from pprint import pformat

writer = SummaryWriter()
# from torch.nn.parallel import DistributedDataParallel as DDP
def log_print(x):
    with open("train.log", "a") as f:
        f.write(str(x))
        f.write("\n")
        print(x)


# In[2]:

N_DIST_UNIT = 240

LAYOUTLM = "microsoft/layoutlm-base-cased"
# BERT = "bert-base-multilingual-cased"
# BERT = LAYOUTLM
BERT = "vinai/phobert-base"
config_bert = AutoConfig.from_pretrained(
    BERT,
    local_files_only=True,
    # max_position_embeddings=1024,
    # hidden_dropout_prob=0.1,
    num_hidden_layers=9,
)
tokenizer = AutoTokenizer.from_pretrained(
    BERT, local_files_only=True, **config_bert.to_dict()
)
config_layoutlm = AutoConfig.from_pretrained(
    LAYOUTLM, local_files_only=True, **config_bert.to_dict()
)


# In[3]:


# dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/train.jsonl")
dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/batch.jsonl")
test_dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/test.jsonl")


# In[4]:


dataloader = DataLoader(dataset, batch_size=1)


# In[5]:


# reload(spade)
print(dataset.nfields)
model = spade.LayoutLMSpade(config=config_bert, num_labels=len(dataset.fields) + 1)
# model = LayoutLMForTokenClassification.from_pretrained(
#     LAYOUTLM, num_labels=len(dataset.fields) + 1
# )
# bert = AutoModel.from_pretrained(BERT)
# model.layoutlm.embeddings.word_embeddings = bert.embeddings.word_embeddings
# model.layoutlm.embeddings.position_embeddings = bert.embeddings.position_embeddings
# checkpoint = os.path.join("checkpoint", list(sorted(os.listdir("checkpoint")))[-1])
# print("Loading checkpoint", checkpoint)
# checkpoint = torch.load(checkpoint)
# model.load_state_dict(checkpoint)
# model.train()
# model = spade.LayoutLMSpade(
#     config_layoutlm,
#     config_bert,
#     LAYOUTLM,
#     BERT,
#     # n_classes=len(["store", "menu", "subtotal", "total", "info"]),
#     fields=dataset.fields,
#     n_classes=dataset.nfields,
#     local_files_only=False,
# )
print(model)
import sys

# sys.exit(1)

# bert = AutoModel.from_pretrained(BERT)
# sd = bert.state_dict()
# for (k, p) in model.backbone.named_parameters():
#     if k in sd and sd[k].shape == p.data.shape:
#         p.data = sd[k]
#         p.require_grad = False
# for p in model.backbone.parameters():
#     p.require_grad = False
# model = DDP(model)
# out = model(dataset[0:4])


# In[6]:


import random
from spade import graph_decoder


# In[7]:


def tail_collision_avoidance(adj, threshold=0.5, lmax=20):
    # adj: m * n matrix
    adj = torch.abs(adj.detach().clone())
    m, n = adj.shape
    old_size = n
    pad_size = m - n
    if pad_size > 0:
        adj = torch.cat([adj, torch.zeros(m, pad_size)], dim=1)
    m, n = adj.shape
    assert m == n

    def node_with_multiple_incoming(adj, threshold):
        nodes = []
        for i in range(n):
            if torch.count_nonzero(adj[:, i] > threshold) > 1:
                nodes.append(i)
        return nodes

    # Iterate in range
    for _ in range(lmax):
        nwmi = node_with_multiple_incoming(adj, threshold)
        if len(nwmi) == 0:
            break

        trimmed_heads = []
        for i in nwmi:
            # j = highest linkink probability node
            j = adj[:, i].argmax()
            pr_ji = adj[j, i]

            # trimmed head nodes
            trimmed_heads.extend(
                [k for k in range(n) if k != j and k != i and adj[k, i] > threshold]
            )

            # trim
            adj[:, i] = 0
            adj[j, i] = 1

        # find new tail nodes
        trimmed_heads = list(set(trimmed_heads))
        for i in trimmed_heads:
            # top 3 nodes with highest probability
            top3_val, top3_idx = torch.topk(adj[i, :], 3)
            new_tails = [j for j in top3_idx if adj[i, j] >= threshold]
            for j in new_tails:
                adj[i, j] = 1
    if pad_size > 0:
        adj = adj[:, :old_size]
    return adj


def true_adj(itc_out, stc_out, idx):
    #     idx = torch.argmin(attention_mask) - 1
    true_itc_out = itc_out[1 : (idx + 1)]
    true_stc_out = stc_out[1 : (idx + 1), 1 : (idx + 1)]
    true_stc_out = torch.sigmoid(true_stc_out)
    return torch.cat([true_itc_out, true_stc_out], dim=-1).transpose(0, 1)


def infer_once(model, dataset, idx=None):
    n = len(dataset)
    if idx is None:
        idx = random.randint(0, n - 1)
    #     idx = 157
    batch = dataset[idx : idx + 1]
    texts = dataset.text_tokens[idx]
    with torch.no_grad():
        # out = model.to("cpu")(batch.to("cpu"))
        b = batch.to("cpu")
        out = model.cpu()(
            input_ids=b.input_ids,
            bbox=b.bbox,
            attention_mask=b.attention_mask,
        )

    # Infer token types
    labels = dataset.fields + ["other", "special"]

    token_predictions = (
        out.logits_c.argmax(-1).squeeze().tolist()
    )  # the predictions are at the token level
    are_box_first_tokens = b.are_box_first_tokens[0].tolist()
    word_level_predictions = []  # let's turn them into word level predictions
    words = []
    token_ids = b.input_ids.squeeze().tolist()

    current_prediction = []
    current_word = []
    for id, token_pred, next_head in zip(
        token_ids, token_predictions, are_box_first_tokens[1:]
    ):
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
        predict_output.append(f"- {word}: {final_label}")

    # Span prediction
    ignore_token_ids = [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    ]
    labels_s = batch.labels_s[0].tolist()
    logits_s = out.logits_s
    span_relation = logits_s.argmax(-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    groups = []
    gt_groups = []
    # for (i, j) in enumerate(span_relation):
    #     if token_ids[i] in ignore_token_ids or token_ids[j] in ignore_token_ids:
    #         continue
    #     if j > 0:
    #         groups.append(f"{tokens[j]} -> {tokens[i]}")
    # for (i, j) in enumerate(labels_s):
    #     if token_ids[i] in ignore_token_ids or token_ids[j] in ignore_token_ids:
    #         continue
    #     if j > 0:
    #         gt_groups.append(f"{tokens[j]} -> {tokens[i]}")

    span_labels = dict(BEGIN=0, INSIDE=1, END=2, SINGLE=3, OTHER=4)
    inv_span_labels = {v: k for (k, v) in span_labels.items()}
    inv_span_labels[None] = "OTHER"
    groups = []
    current_group = []
    for (tk, beiso) in enumerate(span_relation):
        span_label = inv_span_labels[beiso]
        if span_label != "OTHER":
            current_group.append(tokens[tk])
        if span_label == "END":
            groups.append(current_group)
            current_group = []
    gt_groups = []
    current_group = []
    for (tk, beiso) in enumerate(labels_s):
        span_label = inv_span_labels[beiso]
        if span_label != "OTHER":
            current_group.append(tokens[tk])
        if span_label == "END":
            gt_groups.append(current_group)
            current_group = []
    return [predict_output, groups, gt_groups]


# In[8]:


# for p in model.backbone.encoder.parameters():
#     p.require_grad = False


# In[9]:


import transformers


# In[10]:


# opt = torch.optim.Adam(model.parameters(), lr=1e-3)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

# In[11]:


lr_scheduler = transformers.get_cosine_schedule_with_warmup(
    opt, num_warmup_steps=20, num_training_steps=1000
)


# In[ ]:


# In[ ]:


from pprint import pprint


def auto_weight(x):
    x = np.abs(x)
    if x == 0:
        return 1
    return 10 ** (-np.round(np.log10(x)))


device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# loss_labels = ["itc", "rel_s", "rel_g"]
loss_labels = ["itc"]
for e in range(1000):
    total_loss = [0 for _ in loss_labels]
    summary_loss = 0
    for batch_ in tqdm.tqdm(dataloader):
        opt.zero_grad()
        model = model.to(device)
        b = BatchEncoding(batch_).to(device)
        out = model(
            input_ids=b.input_ids,
            bbox=b.bbox,
            attention_mask=b.attention_mask,
            labels_c=b.labels_c,
            labels_s=b.labels_s,
        )
        if not isinstance(out.loss, list):
            loss = out.loss
            total_loss[0] += loss.item()
        elif len(out.loss) > 1:
            weight = [auto_weight(l.item()) for l in out.loss]
            for i, l in enumerate(out.loss):
                lv = l.item()
                if lv == 0:
                    print(f"Warning, loss {i} is zero")
                total_loss[i] += lv
                # l.backward(retain_graph=True)
            # if sum(out.loss).item() > 10:
            loss = sum(out.loss)
            # else:
            #     loss = sum([l * w for (l, w) in zip(out.loss, weight)])
        else:
            loss = out.loss[0]
            total_loss[0] += out.loss[0].item()

        summary_loss += loss.item()
        loss.backward()
        opt.step()
        lr_scheduler.step()
    if e % 50 == 0:
        torch.save(model.state_dict(), "checkpoint/model-%05d.pt" % e)
    try:
        # for i in range(5):
        #     log_print(out.itc_outputs[:, i])
        result = infer_once(model, test_dataset, idx=0)
        # writer.add_text("Infer", "\r".join(result), e)
        log_print(pformat(result))
    except Exception:

        traceback.print_exc()
    log_print(f"epoch {e + 1}")
    nbatch = len(dataloader)
    writer.add_scalar(f"Loss/total", summary_loss, e)
    for (ll, lv) in zip(loss_labels, total_loss):
        log_print(f"     {ll}: {lv}")
        writer.add_scalar(f"Loss/{ll}", lv, e)
    log_print(f"     total: {summary_loss}")


# In[ ]:
