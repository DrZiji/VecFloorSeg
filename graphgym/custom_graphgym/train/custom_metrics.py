import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_metric


@register_metric('edge_classification')
def EdgeClassification(**kwargs):
    aux_pred, aux_true, aux_loss = \
        kwargs['aux_pred'], kwargs['aux_true'], kwargs['aux_loss']
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    true, pred_score = torch.cat(aux_true), torch.cat(aux_pred)
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        pred_int =  (pred_score > cfg.model.thresh).long()
    else:
        pred_int =  pred_score.max(dim=1)[1]  # 返回最大值的index
    try:
        r_a_score = roc_auc_score(true, pred_score)
    except ValueError:
        r_a_score = 0.0

    return {
        'aux_accuracy': round(accuracy_score(true, pred_int), cfg.round),
        'aux_precision': round(precision_score(true, pred_int, pos_label=0), cfg.round),
        'aux_recall': round(recall_score(true, pred_int, pos_label=0), cfg.round),
        'aux_f1': round(f1_score(true, pred_int, pos_label=0), cfg.round),
        'aux_auc': round(r_a_score, cfg.round),
        'aux_loss': sum(aux_loss) / len(aux_loss)
    }
