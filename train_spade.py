#!/usr/bin/env python
# coding: utf-8
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
import os
import tqdm
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from spade.score import scores

writer = SummaryWriter()
# from torch.nn.parallel import DistributedDataParallel as DDP
def log_print(x):
    with open("train.log", "a") as f:
        f.write(str(x))
        f.write("\n")
        print(x)


global BATCH_SIZE
global CHECKPOINTDIR
global NUM_HIDDEN_LAYERS
global LAYOUTLM
global BERT
global train_data
global test_data


def vietnamese():
    global BATCH_SIZE
    global CHECKPOINTDIR
    global NUM_HIDDEN_LAYERS
    global LAYOUTLM
    global BERT
    global train_data
    global test_data

    BATCH_SIZE = 4
    CHECKPOINTDIR = "checkpoint-vnbill"
    NUM_HIDDEN_LAYERS = 0
    LAYOUTLM = "microsoft/layoutlm-base-cased"
    BERT = "vinai/phobert-base"
    train_data = "sample_data/train.jsonl"
    test_data = "sample_data/test.jsonl"


def vietnamese_cccd():
    global BATCH_SIZE
    global CHECKPOINTDIR
    global NUM_HIDDEN_LAYERS
    global LAYOUTLM
    global BERT
    global train_data
    global test_data

    BATCH_SIZE = 4
    CHECKPOINTDIR = "checkpoint-cccd"
    NUM_HIDDEN_LAYERS = 0
    LAYOUTLM = "microsoft/layoutlm-base-cased"
    BERT = "vinai/phobert-base"
    train_data = "sample_data/spade-data/CCCD/train.jsonl"
    test_data = "sample_data/spade-data/CCCD/test.jsonl"

def vietnamese_large():
    global BATCH_SIZE
    global CHECKPOINTDIR
    global NUM_HIDDEN_LAYERS
    global LAYOUTLM
    global BERT
    global train_data
    global test_data

    BATCH_SIZE = 1
    CHECKPOINTDIR = "checkpoint-vnbill-large"
    NUM_HIDDEN_LAYERS = 10
    LAYOUTLM = "microsoft/layoutlm-base-cased"
    BERT = "vinai/phobert-large"
    train_data = "sample_data/train.jsonl"
    test_data = "sample_data/test.jsonl"


def japanese():
    global BATCH_SIZE
    global CHECKPOINTDIR
    global NUM_HIDDEN_LAYERS
    global LAYOUTLM
    global BERT
    global train_data
    global test_data

    LAYOUTLM = "microsoft/layoutlm-base-cased"
    # BERT = "bert-base-multilingual-cased"
    BERT = "cl-tohoku/bert-base-japanese"
    BATCH_SIZE = 3
    CHECKPOINTDIR = "checkpoint-jpcard"
    NUM_HIDDEN_LAYERS = 0
    train_data = "sample_data/spade-data/business_card/train.jsonl"
    test_data = "sample_data/spade-data/business_card/test.jsonl"


# vietnamese_large()
# vietnamese()
# japanese()
vietnamese_cccd()
LOAD_MODEL=True
if NUM_HIDDEN_LAYERS > 0:
    config_bert = AutoConfig.from_pretrained(BERT, num_hidden_layers=NUM_HIDDEN_LAYERS)
else:
    config_bert = AutoConfig.from_pretrained(BERT)
tokenizer = AutoTokenizer.from_pretrained(
    BERT, local_files_only=False, **config_bert.to_dict()
)
config_layoutlm = AutoConfig.from_pretrained(
    LAYOUTLM, local_files_only=True, **config_bert.to_dict()
)


dataset = spade.SpadeDataset(tokenizer, config_bert, train_data)
test_dataset = spade.SpadeDataset(tokenizer, config_bert, test_data)


dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
dataloader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(dataset.nfields)
model = spade.LayoutLMSpade(
    config_layoutlm,
    config_bert,
    fields=dataset.fields,
)
print(model)
# model = nn.DataParallel(model)


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


# In[10]:


# opt = torch.optim.Adam(model.parameters(), lr=5e-5)


# In[11]:


opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
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


try:
    os.mkdir(CHECKPOINTDIR)
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if LOAD_MODEL==True:
    sd = torch.load(f"{CHECKPOINTDIR}/best_score.pt")
            # sd = torch.load("./checkpoint/f1_s_max.pt")
    print(model)
    model.load_state_dict(sd, strict=False)

min_loss = 9999
max_score = 0
loss_labels = ["loss_s", "loss_g"]
s_bests = {}
g_bests = {}
for e in range(1000):
    # if min_loss <= 3:
    #     for p in model.backbone.parameters():
    #         p.require_grad = True
    # else:
    #     for p in model.backbone.parameters():
    #         p.require_grad = False
    total_loss = [0 for _ in loss_labels]
    summary_loss = 0
    for batch_ in tqdm.tqdm(dataloader):
        opt.zero_grad()
        b = BatchEncoding(batch_).to(device)
        model = model.to(device)
        out = model(b)

        if len(out.loss) > 1:
            weight = [auto_weight(l.item()) for l in out.loss]
            for i, l in enumerate(out.loss):
                lv = l.item()
                if lv == 0:
                    print(f"Warning, loss {i} is zero")
                total_loss[i] += lv
                # l.backward(retain_graph=True)
            # if sum(out.loss).item() > 10:
            # loss = sum(out.loss)
            # loss = out.loss[0] * 7 + out.loss[1] * 3
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
        torch.save(model.state_dict(), f"{CHECKPOINTDIR}/model-%05d.pt" % e)
    if loss < min_loss:
        torch.save(model.state_dict(), f"{CHECKPOINTDIR}/model-least-loss.pt")
        min_loss = loss.item()

    try:
        idx = randint(0, len(test_dataset))
        test_batch = test_dataset[idx : idx + 1]
        rel_s, rel_g = test_batch.labels[0, 0], test_batch.labels[0, 1]

        data_id = dataset.raw[idx]["data_id"]
        prediction, rel_s_pr, rel_g_pr = infer_single(
            model, tokenizer, test_dataset, idx
        )
        ground_truth, _ = post_process(
            tokenizer, rel_s, rel_g, test_batch, dataset.fields
        )

        print("---------------------------------")
        print(data_id)
        print("---------------------------------")
        print("Predict:", prediction)
        print("---------------------------------")
        print("GTruth:", ground_truth)
        print("---------------------------------")
        writer.add_text("Inference/prediction", str(prediction), e)
        writer.add_text("Inference/groud_truth", str(ground_truth), e)
    except Exception:
        import traceback

        traceback.print_exc()
        print(data_id)

    for i in range(len(test_dataset)):
        # dataset -> test_dataset...
        classification, rel_s_pr, rel_g_pr = infer_single(
            model, tokenizer, test_dataset, i
        )
        test_batch = test_dataset[i : i + 1]
        rel_s_score_i = scores(test_batch.labels[0][0], rel_s_pr)
        rel_g_score_i = scores(test_batch.labels[0][1], rel_g_pr)
        if i == 0:
            rel_s_score = rel_s_score_i
            rel_g_score = rel_g_score_i
        else:
            for key, value in rel_s_score.items():
                rel_s_score[key] += rel_s_score_i[key]
                rel_g_score[key] += rel_g_score_i[key]

    mean_score_s = rel_s_score
    mean_score_g = rel_g_score
    for key, value in mean_score_s.items():
        mean_score_s[key] = mean_score_s[key] / len(test_dataset)
        mean_score_g[key] = mean_score_g[key] / len(test_dataset)
        pprint(f"Validation_s_edge_{key}: {mean_score_s[key]}")
        pprint(f"Validation_g_edge_{key}: {mean_score_g[key]}")
        writer.add_scalar(f"Score/s_{key}", mean_score_s[key], e)
        writer.add_scalar(f"Score/g_{key}", mean_score_g[key], e)
        if s_bests.get(key, 0) > mean_score_s[key]:
            torch.save(model.state_dict(), f"{CHECKPOINTDIR}/s_best_{key}.pt")
            s_bests[key] = mean_score_s[key]
        if g_bests.get(key, 0) > mean_score_g[key]:
            torch.save(model.state_dict(), f"{CHECKPOINTDIR}/g_best_{key}.pt")
            g_bests[key] = mean_score_s[key]

    current_score = mean_score_s["f1"] + mean_score_g["precision"]
    if current_score > max_score:
        max_score = current_score
        torch.save(model.state_dict(), f"{CHECKPOINTDIR}/best_score.pt")
    log_print(f"epoch {e + 1}")
    nbatch = len(dataloader)
    writer.add_scalar(f"Loss/total", summary_loss, e)
    for (ll, lv) in zip(loss_labels, total_loss):
        log_print(f"     {ll}: {lv}")
        writer.add_scalar(f"Loss/{ll}", lv, e)
    log_print(f"     total: {summary_loss}")
    log_print(f"     best score: {max_score}")
