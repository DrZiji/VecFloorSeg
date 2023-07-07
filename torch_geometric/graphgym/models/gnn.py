import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    BatchNorm1dEdge,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.register import register_stage


def GNNLayer(dim_in, dim_out, has_act=True, **kwargs):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(
            dim_in, dim_out, 1, has_act=cfg.gnn.has_act,
            has_bias=False, has_final_act=cfg.gnn.has_final_act, cfg=cfg
        ), **kwargs
    )


def GNNPreMP(dim_in, dim_out, num_layers, **kwargs):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    # return GeneralMultiLayer(
    #     'linear',
    #     layer_config=new_layer_config(dim_in, dim_out, num_layers,
    #                                   has_act=False, has_bias=False, cfg=cfg))
    return GeneralMultiLayer(
        cfg.gnn.layer_pre_type,
        layer_config=new_layer_config(
            dim_in, dim_out, num_layers,
            has_act=cfg.gnn.pre_has_act, has_bias=False,
            has_final_act=cfg.gnn.pre_has_final_act,
            cfg=cfg
        ), **kwargs
    )

@register_stage('stack')
@register_stage('skipsum')
@register_stage('skipconcat')
class GNNStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out, **kwargs)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif cfg.gnn.stage_type == 'skipconcat' and \
                    i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


class FeatureEncoder(nn.Module):
    """
    Encoding node and edge features

    """
    def __init__(self):
        super().__init__()
        self.dim_out = cfg.share.dim_node_in
        if isinstance(cfg.gnn.dim_inner, list):
            dim_out = cfg.gnn.dim_inner[0]
        else:
            dim_out = cfg.gnn.dim_inner
        # 如果不使用node encoder，就需要通过cfg.share.dim_node_in人工
        # 指定self.dim_out
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(dim_out)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(
                        dim_out, -1, -1, has_act=False,
                        has_bias=False, has_final_act=False, cfg=cfg
                    ))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_out = dim_out
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(dim_out)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(
                    new_layer_config(
                        dim_out, -1, -1, has_act=False,
                        has_bias=False, has_final_act=False, cfg=cfg
                    ))

    def forward(self, batch):
        """"""
        for module in self.children():
            batch = module(batch)
        return batch


class GNN(nn.Module):
    """
    General GNN model: encoder + stage + head

    Args:
        **kwargs (optional): Optional additional args
    """
    def __init__(self, **kwargs):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder()
        dim_in = self.encoder.dim_out

        if cfg.gnn.layer_pre_type == cfg.gnn.layer_type:
            prestageKwargs = dict(cfg.gnn.layer_cfg)
        else:
            prestageKwargs = dict(cfg.gnn.pre_layer_cfg)
        prestageKwargs.update(dict(has_norm_act=False))
        if cfg.gnn.layers_pre_mp > 0:
            if isinstance(cfg.gnn.dim_inner, list):
                dim_out = cfg.gnn.dim_inner[0]
            else:
                dim_out = cfg.gnn.dim_inner
            self.pre_mp = GNNPreMP(
                dim_in, dim_out,
                cfg.gnn.layers_pre_mp, **prestageKwargs
            )
            dim_in = dim_out

        stageKwargs = dict(cfg.gnn.layer_cfg)
        stageKwargs['has_norm_act'] = True
        if cfg.gnn.layers_mp > 0:
            if isinstance(cfg.gnn.dim_inner, list):
                dim_out = cfg.gnn.dim_inner[-1]
            else:
                dim_out = cfg.gnn.dim_inner
            self.mp = GNNStage(
                dim_in=dim_in, dim_out=dim_out,
                num_layers=cfg.gnn.layers_mp, **stageKwargs
            )
            dim_in = dim_out

        headKwargs = dict(cfg.gnn.head_cfg)
        self.post_mp = GNNHead(
            dim_in=dim_in,
            **headKwargs
        )

        self.apply(init_weights)

    def forward(self, batch):

        for module in self.children():
            batch = module(batch)
        return batch
