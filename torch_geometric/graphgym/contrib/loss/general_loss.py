import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.register import register_loss


class M_BCELogitLoss(nn.Module):
    def __init__(self):
        super(M_BCELogitLoss, self).__init__()
        self.to_be_inited = True
        self.loss_term = None

    def init(self, reduction='mean', batch_weight=None, pos_weight=None):
        self.loss_term = nn.BCEWithLogitsLoss(
            weight=batch_weight, pos_weight=pos_weight, reduction=reduction
        )
        self.to_be_inited = False

    def forward(self, inputs, targets):
        assert not self.to_be_inited, 'the loss module should be inited.'
        loss = self.loss_term(inputs, targets)
        return loss


register_loss('binary_cross_entropy', M_BCELogitLoss())