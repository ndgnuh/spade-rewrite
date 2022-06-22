import torch
import numpy as np
# from model_layoutlm import LayoutLMSpade, SpadeDataset
from spade_gcn.model_gnn import SpadeDataset, ProtoGraphConfig, GCNSpade
from traceback import print_exc
from transformers import AutoConfig, AutoTokenizer, BatchEncoding
from torch.utils.data import DataLoader
from pprint import pprint
from functools import cache
from argparse import Namespace
from spade_gcn.score_spade import *
from spade_gcn.score import scores

@cache
def group_name(field):
    if "." in field:
        return field.split(".")[0]
    else:
        return field


def infer_single(model, tokenizer, dataset, i):
    batch = dataset[i:i + 1]
    if isinstance(batch, tuple):
        batch = batch[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model = model.to(device)
        if 'labels' in batch:
            batch.pop('labels')
        batch = BatchEncoding(batch).to(device)
        out = model(batch)
        # rel_s_score = out.rel[0][0, 1].detach()
        # rel_g_score = out.rel[1][0, 1].detach()
        rel_s = out.rel[0].argmax(dim=1)[0].detach().cpu()
        rel_g = out.rel[1].argmax(dim=1)[0].detach().cpu()

    data_id = dataset.raw[i]["data_id"]
    classification, has_loop = post_process(tokenizer, rel_s, rel_g, batch,
                                            dataset.fields)
    return classification, rel_s, rel_g


def force_1d(x):
    if x.dim() == 2:
        return x[0]
    else:
        return x


def post_process(tokenizer, rel_s, rel_g, batch, fields):
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
        current_classification = []
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

                current_classification.append((field, texts))

        if len(current_classification) > 0:
            classification.append(current_classification)

    return classification, has_loop


if __name__ == "__main__":
    # BERT = "vinai/phobert-base"
    # config_bert = AutoConfig.from_pretrained(
    #     BERT,
    #     local_files_only=True,
    #     # max_position_embeddings=1024,
    #     # hidden_dropout_prob=0.1,
    #     num_hidden_layers=9,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(BERT,
    #                                           local_files_only=True,
    #                                           **config_bert.to_dict())
    # config_layoutlm = AutoConfig.from_pretrained(
    #     "microsoft/layoutlm-base-cased",
    #     local_files_only=True,
    #     **config_bert.to_dict())
    # dataset = SpadeDataset(tokenizer, config_bert, "sample_data/test.jsonl")
    # dataloader = DataLoader(dataset, batch_size=1)

    # model = GCNSpade(
    #     config_layoutlm,
    #     config_bert,
    #     "microsoft/layoutlm-base-cased",
    #     BERT,
    #     # n_classes=len(["store", "menu", "subtotal", "total", "info"]),
    #     fields=dataset.fields,
    #     n_classes=dataset.nfields,
    #     local_files_only=False,
    # )

    # try:
    #     sd = torch.load("spade-weights/model-00900.pt")
    #     model.load_state_dict(sd)
    #     for i in range(len(dataset)):
    #         classification = infer_single(model, dataset, i)

    #     # if has_loop:
    #     #     torch.save(
    #     #         {
    #     #             "batch": batch,
    #     #             "rel_s_score": rel_s_score,
    #     #             "rel_g_score": rel_g_score,
    #     #             "rel_s": rel_s,
    #     #             "rel_g": rel_g,
    #     #         },
    #     #         f"loop-{data_id}.pt",
    #     #     )
    #     print()
    #     print("##################################################")
    #     print(dataset.raw[i]["data_id"])
    #     print("##################################################")
    #     pprint(classification)
    # except Exception:
    #     print_exc()


    BERT = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(BERT, local_files_only=False)

    max_epoch = 1000
    MAX_POSITION_EMBEDDINGS = 258 * 3
    OVERLAP = 0
    dataset_config = Namespace(max_position_embeddings=MAX_POSITION_EMBEDDINGS,
                            u_text=30,
                            u_dist=120)
    # dataset = SpadeDataset(tokenizer, dataset_config, train_data)
    test_data=  "../data/vietnamese_invoice_GTGT/dev_invoice_vn.jsonl"
    test_dataset = SpadeDataset(tokenizer, dataset_config, test_data)

    # Because GCN, batch size = 1
    # dataloader = DataLoader(dataset, batch_size=1)
    dataloader_test = DataLoader(test_dataset, batch_size=1)

    s0 = 1
    s1 = 1.5
    s2 = 3
    d_scales = [
        [s0, s0, s0, s0, s0],
        [s0, s0, s1, s1, s1],
        [s1, s1, s1, s1, s1],
        [s1, s1, s2, s2, s2],
        [s2, s2, s2, s2, s2],
        [s2, s2, s0, s0, s0],
    ]
    d_scales = [
        [s0, s0, s0, s1, s1, s1, s2, s2, s2, s0],
    ] * 4
    CHECKPOINTDIR = "checkpoint-gcn-vninvoice-13"
    config = ProtoGraphConfig(
        tokenizer="vinai/phobert-base",
        n_layers=5,
        layer_type="rev_gnn",
        rev_gnn_n_groups=2,
        rev_gnn_mul=True,
        head_rl_layer_type='linear',
        # n_head=12,
        d_model=1280 * 2,
        # d_scales=d_scales,
        # self_loops=[[False] * 4] * 30,
        # update_links=[True] * 30,
        u_text=dataset_config.u_text,
        u_dist=dataset_config.u_dist,
        n_labels=test_dataset.nfields)
    # embeddings = AutoModel.from_pretrained("vinai/phobert-base")
    # embeddings = embeddings.embeddings.word_embeddings
    model = GCNSpade(config)
    print(model)
    # model = nn.DataParallel(model)

    # In[7]:
    

    try:
        sd = torch.load(f"{CHECKPOINTDIR}/best_score_parse.pt", map_location="cpu")
        print(model)
        model.load_state_dict(sd, strict=False)
        f1 = 0
        for i in range(len(test_dataset)):
            classification, rel_s_pr, rel_g_pr = infer_single(
                model, tokenizer, test_dataset, i
            )
            # print(dataset[i])
            # print(dataset[i]["data_id"])

            # import sys
            # sys.exit()
            test_batch, _ = test_dataset[i : i + 1]
            # rel_s_p_r_f1 += cal_p_r_f1_of_edges(test_batch.labels[0][0], rel_s_pr)
            # rel_g_p_r_f1 += cal_p_r_f1_of_edges(test_batch.labels[0][1], rel_g_pr)
            rel_s_score_i = scores(test_batch.labels[0], rel_s_pr)
            rel_g_score_i = scores(test_batch.labels[1], rel_g_pr)
            if i == 0:
                rel_s_score = rel_s_score_i
                rel_g_score = rel_g_score_i
            else:
                for key, value in rel_s_score.items():
                    rel_s_score[key] += rel_s_score_i[key]
                    rel_g_score[key] += rel_g_score_i[key]

            # F1-parse

            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_parse = BatchEncoding(test_batch).to(device)
            pr=post_process_v2(tokenizer, rel_s_pr, rel_g_pr, batch_parse, test_dataset.raw[i]["fields"])      
            gt=post_process_v2(tokenizer, test_batch.labels[0], test_batch.labels[1], batch_parse, test_dataset.raw[i]["fields"])
            if i==0:
                sum_score_parse=score_parse(gt,pr)
            else:
                sum_score_parse+=score_parse(gt,pr)

            ground_truth, _ = post_process(
            tokenizer, test_batch.labels[0], test_batch.labels[1], test_batch, test_dataset.fields
            )
            
            print()
            print("##################################################")
            print(test_dataset.raw[i]["data_id"])
            print("##################################################")
            pprint("GT:")
            pprint(ground_truth)
            print("##################################################")
            pprint("PR:")
            pprint(classification)
            import json
            import shutil
            # data_name=test_dataset.raw[i]["data_id"]
            # with open(f"./output_json/{data_name}.json","w") as f:
            #     shutil.copy(f"../../phung/Anh_Hung/OCR/OCR-invoice/Vietnamese/spade/label_tool/Image_data/invoice_image_27-5/{data_name}",f"./output_json/{data_name}")
            #     json.dump(classification,f,indent=2, ensure_ascii=False)
            print("##################################################")
            print("Validation_f1_parse: ",str(score_parse(gt,pr)))


            

        print("Validation_f1_parse_mean: ", sum_score_parse / len(test_dataset))

        mean_score_s = rel_s_score
        mean_score_g = rel_g_score
        for key, value in mean_score_s.items():
            mean_score_s[key] = mean_score_s[key] / len(test_dataset)
            mean_score_g[key] = mean_score_g[key] / len(test_dataset)
            pprint(f"Validation_s_edge_{key}: {mean_score_s[key]}")
            pprint(f"Validation_g_edge_{key}: {mean_score_g[key]}")

    except Exception:
        print_exc()