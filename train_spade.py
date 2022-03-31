#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding
from dataclasses import dataclass
from typing import Optional
from spade import model_layoutlm as spade
from spade.spade_inference import infer_single, post_process
import numpy as np
from pprint import pformat
from random import randint
import transformers

# from importlib import reload
# reload(spade)
import os
import tqdm
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

writer = SummaryWriter()
# from torch.nn.parallel import DistributedDataParallel as DDP
def log_print(x):
    with open("train.log", "a") as f:
        f.write(str(x))
        f.write("\n")
        print(x)


N_DIST_UNIT = 240

# BERT = "bert-base-multilingual-cased"
# BERT = "bert-base-multilingual-cased"
# BERT = "vinai/phobert-base"
LAYOUTLM = "microsoft/layoutlm-base-cased"
BERT = "cl-tohoku/bert-base-japanese"
BERT = "cl-tohoku/bert-base-japanese"
config_bert = AutoConfig.from_pretrained(BERT)
tokenizer = AutoTokenizer.from_pretrained(
    BERT, local_files_only=False, **config_bert.to_dict()
)
config_layoutlm = AutoConfig.from_pretrained(
    LAYOUTLM, local_files_only=True, **config_bert.to_dict()
)


# train_data = "sample_data/batch.jsonl"
train_data = "sample_data/spade-data/business_card/train.jsonl"
test_data = "sample_data/spade-data/business_card/test.jsonl"
dataset = spade.SpadeDataset(tokenizer, config_bert, train_data)
test_dataset = spade.SpadeDataset(tokenizer, config_bert, test_data)


dataloader = DataLoader(dataset, batch_size=2)


print(dataset.nfields)
model = spade.LayoutLMSpade(
    config_layoutlm,
    config_bert,
    fields=dataset.fields,
)
print(model)


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


# In[10]:


# opt = torch.optim.Adam(model.parameters(), lr=5e-5)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)


# In[11]:


lr_scheduler = transformers.get_cosine_schedule_with_warmup(
    opt, num_warmup_steps=30, num_training_steps=1000
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
loss_labels = ["loss_s", "loss_g"]
for e in range(1000):
    total_loss = [0 for _ in loss_labels]
    summary_loss = 0
    for batch_ in tqdm.tqdm(dataloader):
        opt.zero_grad()
        b = BatchEncoding(batch_).to(device)
        out = model.cuda()(b)

        if len(out.loss) > 1:
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
    if e % 20 == 0:
        torch.save(model.state_dict(), "checkpoint/model-%05d.pt" % e)
    try:
        idx = randint(0, len(test_dataset))
        test_batch = dataset[idx : idx + 1]
        rel_s, rel_g = test_batch.labels[0, 0], test_batch.labels[0, 1]
        data_id = dataset.raw[idx]["data_id"]
        prediction = infer_single(model, tokenizer, test_dataset, idx)
        ground_truth, _ = post_process(
            tokenizer, rel_s, rel_g, test_batch, dataset.fields
        )

        print("---------------------------------")
        print(data_id)
        print("---------------------------------")
        print("Predict:", prediction)
        print("---------------------------------")
        print("GTruth:", ground_truth)
        writer.add_text("Inference/prediction", str(prediction), e)
        writer.add_text("Inference/groud_truth", str(prediction), e)
    except Exception:
        import traceback

        traceback.print_exc()
    log_print(f"epoch {e + 1}")
    nbatch = len(dataloader)
    writer.add_scalar(f"Loss/total", summary_loss, e)
    for (ll, lv) in zip(loss_labels, total_loss):
        log_print(f"     {ll}: {lv}")
        writer.add_scalar(f"Loss/{ll}", lv, e)
    log_print(f"     total: {summary_loss}")
