import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.register import register_loss


class GraphNodeFocalLoss(nn.Module):
    def __init__(self):
        super(GraphNodeFocalLoss, self).__init__()
        self.to_be_inited = True
        self.alpha, self.gamma, self.class_num, self.reduction = None, None, None, None

    def init(self, classes, device, alpha=None, gamma=2, reduction='mean'):
        if alpha is None:
            if classes is None:
                raise RuntimeError('at least one of classes and alpha should be not none.')
            self.alpha = torch.ones(classes, 1)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            elif isinstance(alpha, np.ndarray):
                self.alpha = torch.from_numpy(alpha)
            elif isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                raise RuntimeError('wrong type of alpha param given in focal loss. please check!')
        self.alpha = self.alpha.to(device)
        self.gamma = gamma
        self.class_num = classes
        self.reduction = reduction
        self.to_be_inited = False

    def forward(self, inputs, targets):
        assert not self.to_be_inited, 'the loss module should be inited.'

        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = torch.zeros(
            (N, C), dtype=torch.float32,
            device=inputs.device
        )
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.) # 只保留groundtruth类别的损失

        alpha = self.alpha[targets] # 每一个样本应该乘的权重

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = -1 * alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:
            raise NotImplemented('{} has only implemented loss reduction mean and sum'.format(self))

        return loss


register_loss('focal_loss', GraphNodeFocalLoss())