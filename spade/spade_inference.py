import torch
import numpy as np
from spade.model_layoutlm import LayoutLMSpade, SpadeDataset
from traceback import print_exc
from transformers import AutoConfig, AutoTokenizer, BatchEncoding
from torch.utils.data import DataLoader
from pprint import pprint
from functools import cache


@cache
def group_name(field):
    if "." in field:
        return field.split(".")[0]
    else:
        return field


def infer_single(model, dataset, i):
    batch = dataset[i : i + 1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model = model.to(device)
        batch = BatchEncoding(batch).to(device)
        out = model(batch)
        rel_s_score = out.rel[0][0, 1].detach()
        rel_g_score = out.rel[1][0, 1].detach()
        rel_s = out.rel[0].argmax(dim=1)[0].detach()
        rel_g = out.rel[1].argmax(dim=1)[0].detach()

    data_id = dataset.raw[i]["data_id"]
    classification, has_loop = post_process(
        rel_s, rel_g, batch, dataset.fields, data_id
    )
    return classification


def post_process(rel_s, rel_g, batch, fields, data_id, **kwargs):
    # Convert to numpy because we use matrix indices in a dict
    rel_s = rel_s.cpu().numpy()
    rel_g = rel_g.cpu().numpy()

    nfields = len(fields)
    input_ids = batch.input_ids[0].cpu().tolist()
    input_masks = batch.are_box_first_tokens < 2
    input_masks = input_masks[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    nodes = range(len(fields + input_ids))

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
        current_classification = []
        for i in head_nodes:
            if i not in tails:
                # predicted Rel G may not connect to a head node
                continue
            tail = tails[i]
            input_ids_i = [input_ids[j] for j in tail]
            input_ids_i = [id for id in input_ids_i if id not in ignore_input_ids]
            texts = tokenizer.decode(input_ids_i)
            field = fields[itc[i]]
            # ignore empty fields
            if len(texts) > 0:
                # change dict to tuple for better data structure
                # or change current_classification to dict
                current_classification.append({field: texts})

        if len(current_classification) > 0:
            classification.append(current_classification)

    return classification, has_loop


if __name__ == "__main__":
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
