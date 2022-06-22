import torch
import numpy as np
import math


def ensure_numpy(x):
    # Convert to numpy, or just stay as numpy
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.as_array(x)


def ensure_nz(x):
    # Ensure x is not zero
    return x + 1e-6 if x == 0 else x


def scores(label, pr):
    pr = ensure_numpy(pr)
    label = ensure_numpy(label)
    tp = ((pr == 1) * (label == 1)).sum()
    # tn = ((pr == 0) * (label == 0)).sum()
    fp = ((pr == 1) * (label == 0)).sum()
    fn = ((pr == 0) * (label == 1)).sum()
    return dict(
        # accuracy=(tp + tn) / ensure_nz(tp + tn + fp + fn),
        precision=tp / ensure_nz(tp + fp),
        recall=tp / ensure_nz(tp + fn),
        f1=2 * tp / ensure_nz(2 * tp + fp + fn),
    )


# def cal_p_r_f1_of_edges(labels, pr_labels):
#     """
#     Args:
#         labels: [row, col]
#         pr_labels: [row, col]

#     # tp: true positive
#     # fn: false negative
#     # fp: false positive

#     """
#     idx_gtt = labels == 1  # gt true
#     idx_gtf = labels == 0  # gt false
#     tp = sum(pr_labels[idx_gtt] == 1)
#     fn = sum(pr_labels[idx_gtt] == 0)
#     fp = sum(pr_labels[idx_gtf] == 1)

#     p = tp / ensure_nz(tp + fp)
#     r = tp / ensure_nz(tp + fn)
#     f1 = 2 / ensure_nz(1 / ensure_nz(p) + 1 / ensure_nz(r))
#     # if math.isnan(p) or math.isnan(r) or math.isnan(f1):
#     #     print("tp: ",tp)
#     #     print("fn: ",fn)
#     #     print("fp: ",fp)

#     return torch.tensor([p, r, f1])

# def generate_score_dict(mode, rel_s_p_r_f1, rel_g_p_r_f1):
#     score_dict = {
#         # f"{mode}__avg_loss": avg_loss.item(),
#         # f"{mode}__f1": f1.item(),
#         f"{mode}__precision_edge_avg_s": rel_s_p_r_f1[0].item(),
#         f"{mode}__recall_edge_avg_s": rel_s_p_r_f1[1].item(),
#         f"{mode}__f1_edge_avg_s": rel_s_p_r_f1[2].item(),
#         f"{mode}__precision_edge_avg_g": rel_g_p_r_f1[0].item(),
#         f"{mode}__recall_edge_avg_g": rel_g_p_r_f1[1].item(),
#         f"{mode}__f1_edge_avg_g": rel_g_p_r_f1[2].item(),
#     }

#     return score_dict
