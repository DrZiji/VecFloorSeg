import torch
import torch.nn as nn

from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import DeepGCNLayer, GENConv
from .gate_gen_conv import GateGenConv

@register_layer('deep_gcn+gen_conv')
class DeepGCNwithGEN(nn.Module):
    """
    封装torch_geometric 的 DeepGCNLayer,
    一个layer中集成了dropout, LayerNorm和activation，
    所以不需要GeneralLayer在调用时额外加上上述三种操作
    """
    """
    **kwargs在实现过程中就没有调用，限制的很死
    """
    def __init__(self, layer_config, **kwargs):
        super().__init__()

        #TODO: GENConv的输出特征维度不一定是layer_config.dim_out
        conv = GENConv(
            layer_config.dim_in, layer_config.dim_out, aggr='softmax',
            t=1.0, learn_t=True, num_layers=2, norm='layer'
        )
        norm = nn.LayerNorm(layer_config.dim_in, elementwise_affine=True)
        act = nn.ReLU(inplace=True)
        layer = DeepGCNLayer(
            conv, norm, act, block=layer_config.layer_order, dropout=0.1
        )
        self.model = layer

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            raise NotImplementedError("the GENConv must use edge_attr")
        else:
            batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        return batch


@register_layer('gen_conv')
class GENLayerWraaper(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        conv = GENConv(
            layer_config.dim_in, layer_config.dim_out, aggr='softmax',
            t=1.0, learn_t=True, num_layers=2, norm='layer'
        )
        self.model = conv

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            raise NotImplementedError("the GENConv must use edge_attr")
        else:
            batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        return batch


@register_layer('deep_gcn+gen_conv+xEdgeAttr')
class DeepGCNwithGENXEdge(nn.Module):
    """
    封装torch_geometric 的 DeepGCNLayer,
    一个layer中集成了dropout, LayerNorm和activation，
    所以不需要GeneralLayer在调用时额外加上上述三种操作
    """
    """
    **kwargs在实现过程中就没有调用，限制的很死
    """
    def __init__(self, layer_config, **kwargs):
        super().__init__()

        #TODO: GENConv的输出特征维度不一定是layer_config.dim_out
        conv = GENConv(
            layer_config.dim_in, layer_config.dim_out, aggr='softmax',
            t=1.0, learn_t=True, num_layers=2, norm='layer'
        )
        norm = nn.LayerNorm(layer_config.dim_in, elementwise_affine=True)
        act = nn.ReLU(inplace=True)
        layer = DeepGCNLayer(
            conv, norm, act, block=layer_config.layer_order, dropout=0.1
        )
        self.model = layer

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            raise NotImplementedError("the GENConv must use edge_attr")
        else:
            batch.x = self.model(batch.x, batch.edge_index, None)

        return batch


@register_layer('gen_conv+xEdgeAttr')
class GENLayerWraaperXEdge(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        conv = GENConv(
            layer_config.dim_in, layer_config.dim_out, aggr='softmax',
            t=1.0, learn_t=True, num_layers=2, norm='layer'
        )
        self.model = conv

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            raise NotImplementedError("the GENConv must use edge_attr")
        else:
            batch.x = self.model(batch.x, batch.edge_index, None)

        return batch


@register_layer('deep_gcn_residual_edge_gen_conv')
class DeepGCN_EdgeRes_GEN(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()

        #TODO: GENConv的输出特征维度不一定是layer_config.dim_out
        conv = GENConv(
            layer_config.dim_in, layer_config.dim_out, aggr='softmax',
            t=1.0, learn_t=True, num_layers=2, norm='layer'
        )
        norm = nn.LayerNorm(layer_config.dim_in, elementwise_affine=True)
        act = nn.ReLU(inplace=True)
        layer = DeepGCNLayer(
            conv, norm, act, block=layer_config.layer_order, dropout=0.1
        )
        self.model = layer

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            raise NotImplementedError("the GENConv must use edge_attr")
        else:
            xPholder = torch.zeros(
                [batch.x.shape[0], batch.edge_attr.shape[1]],
                device=batch.x.device, dtype=torch.float
            )
            x = self.model(xPholder, batch.edge_index, batch.edge_attr)
            x_src, x_dst = x[batch.edge_index[0]], x[batch.edge_index[1]]
            batch.edge_attr = batch.edge_attr + x_src + x_dst

        return batch


@register_layer('residual_edge_gen_conv')
class EdgeRes_GENLayerWraaper(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        conv = GENConv(
            layer_config.dim_in, layer_config.dim_out, aggr='softmax',
            t=1.0, learn_t=True, num_layers=2, norm='layer'
        )
        self.model = conv

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            raise NotImplementedError("the GENConv must use edge_attr")
        else:
            xPholder = torch.zeros(
                [batch.x.shape[0], batch.edge_attr.shape[1]],
                device=batch.x.device, dtype=torch.float
            )
            x = self.model(xPholder, batch.edge_index, batch.edge_attr)
            x_src, x_dst = x[batch.edge_index[0]], x[batch.edge_index[1]]
            batch.edge_attr = batch.edge_attr + x_src + x_dst

        return batch
