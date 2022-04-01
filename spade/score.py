import torch
import numpy as np


def cal_p_r_f1_of_edges(labels, pr_labels):
    """
    Args:
        labels: [row, col]
        pr_labels: [row, col]

    # tp: true positive
    # fn: false negative
    # fp: false positive

    """
    idx_gtt = labels == 1  # gt true
    idx_gtf = labels == 0  # gt false
    tp = sum(pr_labels[idx_gtt] == 1)
    fn = sum(pr_labels[idx_gtt] == 0)
    fp = sum(pr_labels[idx_gtf] == 1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 / (1 / p + 1 / r)
    return torch.tensor([p, r, f1])



def generate_score_dict(mode, rel_s_p_r_f1,rel_g_p_r_f1):
    score_dict = {
                # f"{mode}__avg_loss": avg_loss.item(),
                # f"{mode}__f1": f1.item(),
                f"{mode}__precision_edge_avg_s": rel_s_p_r_f1[0].item(),
                f"{mode}__recall_edge_avg_s": rel_s_p_r_f1[1].item(),
                f"{mode}__f1_edge_avg_s": rel_s_p_r_f1[2].item(),
                f"{mode}__precision_edge_avg_g": rel_g_p_r_f1[0].item(),
                f"{mode}__recall_edge_avg_g": rel_g_p_r_f1[1].item(),
                f"{mode}__f1_edge_avg_g": rel_g_p_r_f1[2].item(),
            }

    return score_dict
