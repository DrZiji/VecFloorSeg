import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_head


NormDict = {
    'batch': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
    'group': nn.GroupNorm,
    'instance': nn.InstanceNorm1d,
}


@register_head('node_edge_head')
class NodeEdgeHead(nn.Module):
    """
    GNN prediction head for multi tasks

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, **kwargs):
        super().__init__()
        dim_out = cfg.share.dim_out

        if kwargs.__contains__('norm'):
            self.norm = NormDict[kwargs['norm']](dim_in)
        else:
            self.norm = nn.LayerNorm(dim_in, elementwise_affine=True)
        self.act = nn.ReLU(inplace=True)
        self.layer_post_mp = nn.Linear(dim_in, dim_out)

        if cfg.model.has_aux:
            aux_dim_out = cfg.share.aux_dim_out
            assert aux_dim_out != 0, "wrong auxiliary head dimension"
            # self.aux_norm = nn.LayerNorm(dim_in, elementwise_affine=True)
            if kwargs.__contains__('norm'):
                self.aux_norm = NormDict[kwargs['norm']](dim_in)
            else:
                self.aux_norm = nn.LayerNorm(dim_in, elementwise_affine=True)

            self.aux_linear = nn.Linear(dim_in, aux_dim_out)
            self.aux_head = True

        else:
            self.aux_head = False

        self.NOTWALL = 1
        self.WALL = 0

    def forward(self, batch):
        oriGraph, venoGraph = batch
        venoGraph.x = self.act(self.norm(venoGraph.x))
        venoGraph.x = F.dropout(venoGraph.x, p=0.1, training=self.training)
        venoGraph.x = self.layer_post_mp(venoGraph.x)
        pred, label = venoGraph.x, venoGraph.y

        if self.aux_head:
            vemap_idx = venoGraph.edge_dual_idx.squeeze()
            venoGraph.dual_attr = oriGraph.edge_attr[vemap_idx]
            edgeAttr = self.act(self.aux_norm(venoGraph.dual_attr))
            edgeAttr = F.dropout(edgeAttr, p=0.1, training=self.training)
            edgePred = self.aux_linear(edgeAttr)

            vemap_idx = venoGraph.edge_dual_idx.squeeze()
            vemap_type = venoGraph.edge_type[:, 1].squeeze()
            mask = vemap_type == self.NOTWALL
            edgeLabel = oriGraph.edge_label[vemap_idx]
            # 加入mask，只考虑非墙体的划分，平衡一下类别损失

            # edgeIndex = venoGraph.edge_index.transpose(1, 0)
            # edgeEndAttr1, edgeEndAttr2 = nodeAttr[edgeIndex[:, 0]], nodeAttr[edgeIndex[:, 1]]
            return (pred, edgePred), (label, edgeLabel)

        else:
            return pred, label

    def forward_test(self, batch):
        oriGraph, venoGraph = batch
        venoGraph.x = self.act(self.norm(venoGraph.x))
        venoGraph.x = F.dropout(venoGraph.x, p=0.1, training=self.training)
        return oriGraph, venoGraph



@register_head('node_edge_head_1')
class NodeEdgeHead1(NodeEdgeHead):
    # 利用venoGraph的edge attr进行aux head分类
    def __init__(self, dim_in, **kwargs):
        super().__init__(dim_in, **kwargs)

    def forward(self, batch):
        oriGraph, venoGraph = batch
        venoGraph.x = self.act(self.norm(venoGraph.x))
        venoGraph.x = F.dropout(venoGraph.x, p=0.1, training=self.training)
        venoGraph.x = self.layer_post_mp(venoGraph.x)
        pred, label = venoGraph.x, venoGraph.y

        if self.aux_head:
            edgeAttr = self.act(self.aux_norm(venoGraph.edge_attr))
            edgeAttr = F.dropout(edgeAttr, p=0.1, training=self.training)
            edgePred = self.aux_linear(edgeAttr)

            vemap_idx = venoGraph.edge_dual_idx.squeeze()
            vemap_type = venoGraph.edge_type[:, 1].squeeze()
            mask = vemap_type == self.NOTWALL
            edgeLabel = oriGraph.edge_label[vemap_idx]
            # 加入mask，只考虑非墙体的划分，平衡一下类别损失
            return (pred, edgePred), (label, edgeLabel)

        else:
            return pred, label


@register_head('node_edge_head_2')
class NodeEdgeHead2(NodeEdgeHead):
    # 利用oriGraph的edge attr进行aux head分类
    def __init__(self, dim_in, **kwargs):
        super().__init__(dim_in, **kwargs)

    def forward(self, batch):
        oriGraph, venoGraph = batch
        venoGraph.x = self.act(self.norm(venoGraph.x))
        venoGraph.x = F.dropout(venoGraph.x, p=0.1, training=self.training)
        venoGraph.x = self.layer_post_mp(venoGraph.x)
        pred, label = venoGraph.x, venoGraph.y

        if self.aux_head:
            edgeAttr = self.act(self.aux_norm(oriGraph.edge_attr))
            edgeAttr = F.dropout(edgeAttr, p=0.1, training=self.training)
            edgePred = self.aux_linear(edgeAttr)

            edgeLabel = oriGraph.edge_label
            # 加入mask，只考虑非墙体的划分，平衡一下类别损失
            return (pred, edgePred), (label, edgeLabel)

        else:
            return pred, label