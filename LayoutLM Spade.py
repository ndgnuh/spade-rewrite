#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding
from dataclasses import dataclass
from typing import Optional
from spade import model_layoutlm as spade
import numpy as np

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


# In[2]:

N_DIST_UNIT = 240

# BERT = "bert-base-multilingual-cased"
# BERT = "bert-base-multilingual-cased"
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
    "microsoft/layoutlm-base-cased", local_files_only=True, **config_bert.to_dict()
)


# In[3]:


dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/train.jsonl")
# dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/batch.jsonl")
test_dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/test.jsonl")


# In[4]:


dataloader = DataLoader(dataset, batch_size=2)


# In[5]:


# reload(spade)
print(dataset.nfields)
model = spade.LayoutLMSpade(
    config_layoutlm,
    config_bert,
    "microsoft/layoutlm-base-cased",
    BERT,
    # n_classes=len(["store", "menu", "subtotal", "total", "info"]),
    fields=dataset.fields,
    n_classes=dataset.nfields,
    local_files_only=False,
)
# sd = torch.load("checkpoint/model-00500.pt")
# model.load_state_dict(sd)
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
    with torch.no_grad():
        out = model.to("cpu")(batch.to("cpu"))

    ignore = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]
    input_ids = batch.input_ids[0].tolist()
    texts = tokenizer.convert_ids_to_tokens(input_ids)
    rel_g = out.rel[1].argmax(dim=1)[0]
    threshold = 0.5
    print(rel_g)

    # rel_s = tail_collision_avoidance(rel_s, threshold)
    # rel_g = tail_collision_avoidance(rel_g, threshold)

    # rel_s = rel_s.numpy() > threshold
    # rel_g = rel_g.numpy() > threshold
    nodes = dataset.fields + texts

    # print(rel_s.shape, len(nodes), len(texts))

    #     print("------+ rel s +---------")
    #     edge_s = []
    edge_g = []
    ignore = [tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id]
    ignore_text = dataset.fields + [
        tokenizer.convert_ids_to_tokens(id) for id in ignore
    ]
    for i, j in zip(*torch.where(rel_g)):
        try:
            node_i = nodes[i]
            node_j = texts[j]
            if node_i in ignore_text or node_j in ignore_text:
                continue

            edge_g.append((node_i, node_j))
        except Exception as e:
            import traceback

            traceback.print_exc()
            pass
    # nfields = len(dataset.fields)
    rel_g_gt = batch.labels[0, 1]
    edge_g_gt = []
    for i, j in zip(*torch.where(rel_g_gt)):
        try:
            node_i = nodes[i]
            node_j = texts[j]
            if node_i in ignore_text or node_j in ignore_text:
                continue

            edge_g_gt.append((node_i, node_j))
        except Exception as e:
            import traceback

            traceback.print_exc()
            pass

    return [edge_g_gt, edge_g]


# In[8]:


# for p in model.backbone.encoder.parameters():
#     p.require_grad = False


# In[9]:


import transformers


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
        # for i in range(5):
        #     log_print(out.itc_outputs[:, i])
        result = infer_once(model, test_dataset, 0)
        pprint(result)
        # writer.add_text("Infer", "\r".join(result[:6]), e)
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


# In[ ]:
