import torch
from transformers import AutoTokenizer
from pprint import pprint
import re
from collections import Counter
from fuzzywuzzy import fuzz
import numpy as np


def force_1d(x):
    if x.dim() == 2:
        return x[0]
    else:
        return x


# post_process
def post_process_v2(tokenizer, rel_s, rel_g, batch, fields):
    # Convert to numpy because we use matrix indices in a dict
    rel_s = rel_s.cpu().numpy()
    rel_g = rel_g.cpu().numpy()
    m, n = rel_s.shape

    nfields = len(fields)
    for i in range(n):
        rel_s[i + nfields, i] = 0
        rel_g[i + nfields, i] = 0

    if 'token_edge_index' in batch:
        tki = batch['token_edge_index']
        for (i, j) in zip(tki[0].tolist(), tki[1].tolist()):
            rel_s[i + nfields, j] = 1

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
                # replace unfinised token
                # single value fields:
                single_keywords = [
                    'total', 'unit', 'rate', '.tax', 'rate', 'vat'
                    'menu.price', 'menu.quantity', 'menu.id', 'menu.count'
                ]
                is_single = any(
                    [keyword in field for keyword in single_keywords])
                if is_single:
                    texts = texts.split(" ")[0]
                # change dict to tuple for better data structure
                # or change current_classification to dict

                # current_classification.append((field, texts))
                current_classification[field] = [texts]

        if len(current_classification) > 0:
            classification.append(current_classification)

    return classification


def group_name(field):
    if "." in field:
        return field.split(".")[0]
    else:
        return field


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


def norm_receipt(val, key):
    val = val.replace(" ", "")
    return val


def get_scores(tp, fp, fn):
    pr = tp / (tp + fp) if (tp + fp) != 0 else 0
    re = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * pr * re) / (pr + re) if (pr + re) != 0 else 0
    return pr, re, f1


def score_parse(gt, pr):
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
            pr_val = ([norm_receipt(val, key)
                       for val in pr[j][key]] if key in pr[j] else [])
            gt_val = [norm_receipt(val, key) for val in gt[i][key]]
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


# def get_price_from_parse(parse):
#     price = 0
#     total = 0
#     for gr in parse:
#         for key in gr.keys():
#             val = get_int_number(gr[key][0].split(" ")[0])
#             if "menu.price" == key or "menu.sub_price" == key:
#                 price += val
#             elif "menu.discountprice" == key:
#                 price -= val
#             elif "total.total_price" == key:
#                 total += val

#     return price, total
# def get_int_number(string):

#     num = re.sub(r"[^0-9]", "", string)
#     return int(num) if len(num) > 0 else 0

# def get_stats(gt,pr):
#     stats= {'label_stats': {}, 'group_stats': [0, 0, 0], 'receipt_cnt': 0, 'price_count_cnt': 0, 'prices_cnt': 0, 'receipt_total': 0}
#     label_stats = stats["label_stats"]
#     group_stats = stats["group_stats"]

#     receipt_refine=False,
#     receipt_edit_distance=False
#     return_refined_parses=False

#     mat = np.zeros((len(gt), len(pr)), dtype=np.int)
#     for i, gr1 in enumerate(gt):
#         for j, gr2 in enumerate(pr):
#             mat[i][j] = get_group_compare_score(gr1, gr2)

#     pairs = []
#     for _ in range(min(len(gt), len(pr))):
#         if np.max(mat) == 0:
#             break

#         x = np.argmax(mat)
#         y = int(x / len(pr))
#         x = int(x % len(pr))
#         mat[y, :] = 0
#         mat[:, x] = 0
#         pairs.append((y, x))
#     for i in range(len(gt)):
#         stat = dict()
#         for key in gt[i]:
#             if key not in stat:
#                 stat[key] = 0
#             stat[key] += 1

#         for key in stat:
#             if key not in label_stats:
#                 label_stats[key] = [0, 0, 0]
#             label_stats[key][1] += stat[key]

#     for i in range(len(pr)):
#         stat = dict()
#         for key in pr[i]:
#             if key not in stat:
#                 stat[key] = 0
#             stat[key] += 1

#         for key in stat:
#             if key not in label_stats:
#                 label_stats[key] = [0, 0, 0]
#             label_stats[key][2] += stat[key]

#     group_stat = [0, len(gt), len(pr)]
#     price_count_check = True
#     for i, j in pairs:
#         # For each group,
#         stat = dict()
#         for key in set(list(gt[i].keys()) + list(pr[j].keys())):
#             if key not in stat:
#                 stat[key] = 0

#         cnt = 0
#         for key in gt[i]:
#             pr_val = (
#                 [norm_receipt(val, key) for val in pr[j][key]] if key in pr[j] else []
#             )
#             gt_val = [norm_receipt(val, key) for val in gt[i][key]]

#             # if key in pr[j] and pr[j][key] == gt[i][key]:
#             #    stat[key] += 1
#             #    cnt += 1
#             if pr_val == gt_val:
#                 stat[key] += 1
#                 cnt += 1

#         if cnt == len(gt[i]):
#             group_stat[0] += 1

#         # Stat Update
#         for key in stat:
#             if key not in label_stats:
#                 label_stats[key] = [0, 0, 0]
#             label_stats[key][0] += stat[key]

#     for k in range(3):
#         group_stats[k] += group_stat[k]

#     label_stats["total"] = [0, 0, 0]
#     for key in sorted(label_stats):
#         if key not in ["total"]:
#             for i in range(3):
#                 label_stats["total"][i] += label_stats[key][i]

#     stats["label_stats"] = label_stats
#     stats["group_stats"] = group_stats

#     return stats
#     # print(stats)

# def summary_receipt(stats, print_screen=False):
#     st = stats["label_stats"]
#     st["Group_accuracy"] = stats["group_stats"]

#     # print("stats[label_stats]: ",st)
#     # print("stats[group_stats]: ",stats["group_stats"])

#     s = dict()
#     for key in st:
#         tp = st[key][0]
#         fp = st[key][2] - tp
#         fn = st[key][1] - tp
#         s[key] = (tp, fp, fn) + get_scores(tp, fp, fn)

#     # print("s: ",s)
#     c = {
#         "main_key": "receipt",
#         "prices": stats["prices_cnt"],
#         "price/cnt": stats["price_count_cnt"],
#         "receipt": stats["receipt_cnt"],
#         "total": stats["receipt_total"],
#     }

#     if print_screen:
#         other_fields = ["total", "Group_accuracy"]
#         header = ("field", "tp", "fp", "fn", "prec", "rec", "f1")
#         print("%25s\t%6s\t%6s\t%6s\t%6s\t%6s\t%6s" % header)
#         print(
#             "------------------------------------------------------------------------------"
#         )
#         for key in sorted(s):
#             if key not in other_fields:
#                 print(
#                     "%-25s\t%6d\t%6d\t%6d\t%6.3f\t%6.3f\t%6.3f"
#                     % (
#                         key,
#                         s[key][0],
#                         s[key][1],
#                         s[key][2],
#                         s[key][3],
#                         s[key][4],
#                         s[key][5],
#                     )
#                 )
#         print(
#             "------------------------------------------------------------------------------"
#         )
#         for key in other_fields:
#             print(
#                 "%-25s\t%6d\t%6d\t%6d\t%6.3f\t%6.3f\t%6.3f"
#                 % (
#                     key,
#                     s[key][0],
#                     s[key][1],
#                     s[key][2],
#                     s[key][3],
#                     s[key][4],
#                     s[key][5],
#                 )
#             )

#         for key in c:
#             if key not in ["total", "main_key"]:
#                 print(
#                     " - %10s accuracy :  %.4f (%d/%d)"
#                     % (key, c[key] / c["total"], c[key], c["total"])
#                 )

#     return s, c
