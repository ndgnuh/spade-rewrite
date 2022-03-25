#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding
from dataclasses import dataclass
from typing import Optional
from spade import model_layoutlm as spade

# from importlib import reload
# reload(spade)
import os
import tqdm
import json
import torch
from torch.utils.data import Dataset, DataLoader

# from torch.nn.parallel import DistributedDataParallel as DDP
def log_print(x):
    with open("train.log", "a") as f:
        f.write(str(x))
        f.write("\n")
        pprint(x)


# In[2]:


# BERT = "bert-base-multilingual-cased"
# BERT = "bert-base-multilingual-cased"
BERT = "vinai/phobert-base"
config_bert = AutoConfig.from_pretrained(
    BERT,
    local_files_only=True,
    # max_position_embeddings=1024,
    # hidden_dropout_prob=0.1,
    # num_hidden_layers=5,
)
tokenizer = AutoTokenizer.from_pretrained(
    BERT, local_files_only=True, **config_bert.to_dict()
)
config_layoutlm = AutoConfig.from_pretrained(
    "microsoft/layoutlm-base-cased", local_files_only=True, **config_bert.to_dict()
)


# In[3]:


dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/train.jsonl")
test_dataset = spade.SpadeDataset(tokenizer, config_bert, "sample_data/test.jsonl")


# In[4]:


dataloader = DataLoader(dataset, batch_size=4)


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
    local_files_only=True,
)
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
        out = model.to("cpu")(batch.to("cpu"))

    #     print(idx)
    #     print("stc_outputs", out.stc_outputs.shape)
    #     print("itc_outputs", out.itc_outputs.shape)
    #     print("attention_mask", out.attention_mask.shape)
    rel_s = out.rel[0, 0]
    print(rel_g.shape)
    rel_g = out.rel[:, 0]
    # rel_s = true_adj(out.itc_outputs[0], out.stc_outputs[0, 0, :, :], len(texts))
    # rel_g = true_adj(
    #     torch.zeros(out.itc_outputs[0].shape), out.stc_outputs[1, 0, :, :], len(texts)
    # )
    threshold = 0.5
    rel_s = tail_collision_avoidance(rel_s, threshold)
    rel_g = tail_collision_avoidance(rel_g, threshold)
    # print('rel_s', rel_s.shape)
    #     print('rel_g', rel_s.shape)
    #     print('texts', texts)
    parsed = graph_decoder.parse_graph(
        [rel_s >= threshold, rel_g >= threshold],
        texts=texts,
        fields=dataset.fields,
        strict=False,
    )
    # for p in parsed:
    #     for (k, v) in p.items():
    #         ids = tokenizer.convert_tokens_to_ids(v.split())
    #         p[k] = tokenizer.decode(ids)

    return parsed


# In[8]:


# for p in model.backbone.encoder.parameters():
#     p.require_grad = False


# In[9]:


import transformers


# In[10]:


opt = torch.optim.Adam(model.parameters(), lr=7e-4)


# In[11]:


lr_scheduler = transformers.get_cosine_schedule_with_warmup(
    opt, num_warmup_steps=1, num_training_steps=2000
)


# In[ ]:


# In[ ]:


from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
for e in range(2000):
    total_loss = 0
    total_itc_loss = 0
    total_stc_loss = 0
    for batch_ in tqdm.tqdm(dataloader):
        opt.zero_grad()
        b = BatchEncoding(batch_).to(device)
        out = model.cuda()(b)
        loss = out.loss
        # itc_loss, stc_loss = out.loss
        # total_itc_loss += itc_loss.item()
        # total_stc_loss += stc_loss.item()
        # itc_loss.backward()
        # loss = 2 * (itc_loss * stc_loss) / (itc_loss + stc_loss)
        # total_loss += loss.item()
        loss.backward()
        opt.step()
        lr_scheduler.step()
    log_print(f"epoch {e + 1} itc_loss: {total_itc_loss / len(dataloader)}")
    log_print(f"epoch {e + 1} stc_loss: {total_stc_loss / len(dataloader)}")
    # log_print(f"epoch {e + 1} loss: {total_loss / len(dataloader)}")
    if e % 50 == 0:
        torch.save(model.state_dict(), "checkpoint/model-%05d.pt" % e)
    try:
        for i in range(5):
            log_print(out.itc_outputs[:, i])
        log_print(infer_once(model, test_dataset, 0))
    except Exception:
        import traceback

        traceback.print_exc()


# In[ ]:
