import torch
import numpy as np
from spade.model_layoutlm import LayoutLMSpade, SpadeDataset
from traceback import print_exc
from transformers import AutoConfig, AutoTokenizer, BatchEncoding
from torch.utils.data import DataLoader
from pprint import pprint
from functools import cache
from collections import Counter


@cache
def group_name(field):
    if "." in field:
        return field.split(".")[0]
    else:
        return field


def force_1d(x):
    if x.dim() == 2:
        return x[0]
    else:
        return x


def post_process(tokenizer, rel_s, rel_g, batch, fields):
    # Convert to numpy because we use matrix indices in a dict
    rel_s = rel_s.cpu().numpy()
    rel_g = rel_g.cpu().numpy()

    nfields = len(fields)
    input_ids = force_1d(batch.input_ids).cpu().tolist()
    input_masks = batch.are_box_first_tokens < 2
    input_masks = force_1d(input_masks).cpu().tolist()

    itc_rel = rel_s[:nfields, :]
    itc = {}
    for (i, j) in zip(*np.where(itc_rel)):
        itc[j] = i

    # subsequence token classification
    visited = {}
    tails = {i: [] for i in itc}

    # tail nodes
    has_loop = False

    def visit(i, head, depth=0):
        if depth >= 100:
            return
        if visited.get(i, False):
            return
        visited[i] = True
        tails[head].append(i)
        for j in np.where(rel_s[i + nfields, :])[0]:
            visit(j, head=head, depth=depth + 1)

    for i in itc:
        visit(i, i)

    # groups head nodes
    groups = {}
    has_group = {i: False for i in itc}
    for (i, j) in zip(*np.where(rel_g[nfields:, :])):
        # print(i, j, tokens[i], tokens[j])
        if i not in groups:
            groups[i] = [i]
            has_group[i] = True
        groups[i].append(j)
        has_group[j] = True

    # each group contain the head nodes
    groups = list(groups.values())

    # Standalone label which doesn't have group
    standalones = {}
    for (i, field_id) in itc.items():
        if has_group[i]:
            continue
        g = group_name(fields[field_id])
        standalone = standalones.get(g, [])
        standalone.append(i)
        standalones[g] = standalone
    standalones = [l for l in standalones.values() if len(l) > 0]
    groups = standalones + groups

    # Tokenizer and add label to each group
    classification = []
    ignore_input_ids = [
        tokenizer.pad_token_id,
        tokenizer.sep_token_id,
        tokenizer.cls_token_id,
    ]
    for head_nodes in groups:
        # Get the tails tokens and concat them to full text
        # current_classification = []
        current_classification = {}
        for i in head_nodes:
            if i not in tails:
                # predicted Rel G may not connect to a head node
                continue
            tail = tails[i]
            input_ids_i = [input_ids[j] for j in tail]
            input_ids_i = [
                id for id in input_ids_i if id not in ignore_input_ids
            ]
            texts = tokenizer.decode(input_ids_i)
            field = fields[itc[i]]
            # ignore empty fields
            if len(texts) > 0:
                # change dict to tuple for better data structure
                # or change current_classification to dict

                # current_classification.append((field, texts))
                current_classification[field] = [texts]

        if len(current_classification) > 0:
            classification.append(current_classification)

    return classification


if __name__ == "__main__":
    BERT = "vinai/phobert-base"
    config_bert = AutoConfig.from_pretrained(
        BERT,
        local_files_only=True,
        # max_position_embeddings=1024,
        # hidden_dropout_prob=0.1,
        num_hidden_layers=9,
    )
    tokenizer = AutoTokenizer.from_pretrained(BERT,
                                              local_files_only=True,
                                              **config_bert.to_dict())
    config_layoutlm = AutoConfig.from_pretrained(
        "microsoft/layoutlm-base-cased",
        local_files_only=True,
        **config_bert.to_dict())
    dataset = SpadeDataset(tokenizer, config_bert, "sample_data/test.jsonl")
    dataloader = DataLoader(dataset, batch_size=1)

    model = LayoutLMSpade(
        config_layoutlm,
        config_bert,
        "microsoft/layoutlm-base-cased",
        BERT,
        # n_classes=len(["store", "menu", "subtotal", "total", "info"]),
        fields=dataset.fields,
        n_classes=dataset.nfields,
        local_files_only=False,
    )

    try:
        sd = torch.load("spade-weights/model-00900.pt")
        model.load_state_dict(sd)
        for i in range(len(dataset)):
            classification = infer_single(model, dataset, i)

        # if has_loop:
        #     torch.save(
        #         {
        #             "batch": batch,
        #             "rel_s_score": rel_s_score,
        #             "rel_g_score": rel_g_score,
        #             "rel_s": rel_s,
        #             "rel_g": rel_g,
        #         },
        #         f"loop-{data_id}.pt",
        #     )
        print()
        print("##################################################")
        print(dataset.raw[i]["data_id"])
        print("##################################################")
        pprint(classification)
    except Exception:
        print_exc()


def group_counter(group):
    txt = []
    for key in group:
        txt = txt + group[key]
    return Counter(sorted("".join(txt).replace(" ", "")))


def get_group_compare_score(gr1, gr2):
    score = 0
    # if gr1.keys() == gr2.keys():
    #    score += 100

    for key in list(set(list(gr1.keys()) + list(gr2.keys()))):
        if key in gr1 and key in gr2:
            # score+=fuzz.ratio(gr1[key],gr2[key])
            if gr1[key] == gr2[key]:
                score += 50
            elif gr1[key] in gr2[key] or gr2[key] in gr1[key]:
                score += 30
            else:
                score += sum((Counter("".join(gr1[key]))
                              & Counter("".join(gr2[key]))).values())

    score += sum((group_counter(gr1) & group_counter(gr2)).values())
    return score


def get_scores(tp, fp, fn):
    pr = tp / (tp + fp) if (tp + fp) != 0 else 0
    re = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * pr * re) / (pr + re) if (pr + re) != 0 else 0
    return pr, re, f1


def get_parse_score(gt, pr):
    label_stats = {}
    # Ma trận tính điểm giữa pr và gt
    mat = np.zeros((len(gt), len(pr)), dtype=np.int)
    for i, gr1 in enumerate(gt):

        for j, gr2 in enumerate(pr):
            mat[i][j] = get_group_compare_score(gr1, gr2)

    # Trích ra cặp pr, gt tương ứng
    pairs = []
    for _ in range(min(len(gt), len(pr))):
        if np.max(mat) == 0:
            break

        x = np.argmax(mat)
        y = int(x / len(pr))
        x = int(x % len(pr))
        mat[y, :] = 0
        mat[:, x] = 0
        pairs.append((y, x))

    # Tính
    for i in range(len(gt)):
        stat = dict()
        for key in gt[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][1] += stat[key]
    for i in range(len(pr)):
        stat = dict()
        for key in pr[i]:
            if key not in stat:
                stat[key] = 0
            stat[key] += 1

        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][2] += stat[key]

    for i, j in pairs:
        # For each group,
        stat = dict()
        for key in set(list(gt[i].keys()) + list(pr[j].keys())):
            if key not in stat:
                stat[key] = 0

        cnt = 0
        for key in gt[i]:
            pr_val = ([val.replace(" ", "")
                       for val in pr[j][key]] if key in pr[j] else [])
            gt_val = [val.replace(" ", "") for val in gt[i][key]]
            if pr_val == gt_val:
                stat[key] += 1
                cnt += 1
        # Stat Update
        for key in stat:
            if key not in label_stats:
                label_stats[key] = [0, 0, 0]
            label_stats[key][0] += stat[key]

    label_stats["total"] = [0, 0, 0]
    for key in sorted(label_stats):
        if key not in ["total"]:
            for i in range(3):
                label_stats["total"][i] += label_stats[key][i]

    s = dict()
    for key in label_stats:
        tp = label_stats[key][0]
        fp = label_stats[key][2] - tp
        fn = label_stats[key][1] - tp
        s[key] = (tp, fp, fn) + get_scores(tp, fp, fn)

    return s["total"][5]
