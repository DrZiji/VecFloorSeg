import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module

from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.register import register_network


@register_network('CUBICASA')
class CubicasaGNN(GNN):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        fplanVertexGraph, fplanVenoGraph = \
            batch[0], batch[1]
        res = super().forward(fplanVenoGraph)
        return res


@register_network('CUBICASA_EDGE')
class CubicasaEdgeGNN(GNN):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        fplanVertexGraph, fplanVenoGraph = \
            batch[0], batch[1]
        res = super().forward(fplanVertexGraph)
        return res

