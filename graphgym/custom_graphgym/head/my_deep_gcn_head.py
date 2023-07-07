import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_head


@register_head('deep_gcn_node')
class DeepGCNNodeHead(nn.Module):
    """
    GNN prediction head for node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in, elementwise_affine=True)
        self.act = nn.ReLU(inplace=True)
        # self.layer_post_mp = MLP(
        #     new_layer_config(
        #         dim_in, dim_out, cfg.gnn.layers_post_mp,
        #         has_act=False, has_bias=True, has_final_act=False,
        #         cfg=cfg
        #     ))
        self.layer_post_mp = nn.Linear(dim_in, cfg.share.dim_out)


    def forward(self, batch):
        batch.x = self.act(self.norm(batch.x))
        batch.x = F.dropout(batch.x, p=0.1, training=self.training)
        batch.x = self.layer_post_mp(batch.x)
        pred, label = batch.x, batch.y
        return pred, label


@register_head('deep_gcn_edge')
class DeepGCNEdgeHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in, elementwise_affine=True)
        self.act = nn.ReLU(inplace=True)
        self.layer_post_mp = nn.Linear(dim_in, dim_out)
        self.NOTWALLTYPE = 1

    def forward(self, batch):
        batch.x = self.act(self.norm(batch.edge_attr))
        batch.x = F.dropout(batch.edge_attr, p=0.1, training=self.training)
        batch.edge_attr = self.layer_post_mp(batch.edge_attr)
        pred, label = batch.edge_attr, batch.edge_label

        maskedPred = pred[batch.edge_type[:, 1] == self.NOTWALLTYPE]
        maskedLabel = label[batch.edge_type[:, 1] == self.NOTWALLTYPE]
        return maskedPred, maskedLabel