import torch

from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.graphgym.config import cfg


"""尝试使用位置向量作为图节点和边的特征"""
@register_node_encoder('sin_pos_node')
class SinePosNodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super(SinePosNodeEncoder, self).__init__()
        self.model = torch.nn.Linear(
            cfg.share.dim_node_in, embed_dim
        )
        # self.norm = torch.nn.LayerNorm(
        #     embed_dim, elementwise_affine=True
        # )

    def forward(self, batch):
        batch.x = self.model(batch.x)
        return batch


@register_edge_encoder('sin_pos_edge')
class SinePosEdgeEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super(SinePosEdgeEncoder, self).__init__()
        self.model = torch.nn.Linear(
            cfg.share.dim_edge_in, embed_dim
        )
        # self.norm = torch.nn.LayerNorm(
        #     embed_dim, elementwise_affine=True
        # )

    def forward(self, batch):
        batch.edge_attr = self.model(batch.edge_attr)
        return batch


@register_edge_encoder('sin_pos_t_edge')
class SinePosEbdEdgeEncoder(torch.nn.Module):
    # 单独使用nn.embedding作为edge attr
    def __init__(self, embed_dim):
        super(SinePosEbdEdgeEncoder, self).__init__()
        self.model = torch.nn.Linear(
            cfg.share.dim_edge * cfg.share.dim_pos, embed_dim
        )
        self.type_model = torch.nn.Linear(
            cfg.share.dim_pos, embed_dim
        )
        self.type_weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, batch):
        # 是否需要根据edge_type调整edge的特征情况?
        # 这一点需要根据边特征是否给模型的分割提供很大帮助才能决定
        edge_type = self.type_model(batch.edge_type)
        batch.edge_attr = edge_type
        return batch


@register_edge_encoder('sin_pos_t_edge_2')
class SinePosOneHotEdgeEncoder(torch.nn.Module):
    # 单独使用one-hot作为edge type
    def __init__(self, embed_dim):
        super(SinePosOneHotEdgeEncoder, self).__init__()
        self.model = torch.nn.Linear(
            cfg.share.dim_edge * cfg.share.dim_pos, embed_dim
        )
        self.type_model = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, embed_dim)
        )

    def forward(self, batch):
        edge_type = self.type_model(batch.edge_type)
        batch.edge_attr = edge_type
        return batch


@register_edge_encoder('sin_pos_t_edge_1')
class SinePosBothEdgeEncoder(torch.nn.Module):
    # 同时使用one-hot的edge type，和edge feature相加作为edge attr
    def __init__(self, embed_dim):
        super(SinePosBothEdgeEncoder, self).__init__()
        self.model = torch.nn.Linear(
            cfg.share.dim_edge * cfg.share.dim_pos, embed_dim
        )
        self.type_model = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, embed_dim)
        )
        self.type_weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, batch):
        edge_attr = self.model(batch.edge_attr)
        edge_type = self.type_model(batch.edge_type)
        batch.edge_attr = edge_attr + edge_type * self.type_weight
        return batch


