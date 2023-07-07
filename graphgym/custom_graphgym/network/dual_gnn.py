import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNN, GNNLayer
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.register import register_stage


@register_stage('dual_stack')
@register_stage('dual_skipconcat')
@register_stage('dual_skipsum')
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
            if cfg.gnn.stage_type == 'dual_skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out, **kwargs)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            xs = [b.x for b in batch]
            batch = layer(batch)

            temp = []
            batch = list(batch)
            for idx, x in enumerate(xs):
                b = batch[idx]
                if cfg.gnn.stage_type == 'dual_skipsum':
                    b.x = x + b.x
                elif cfg.gnn.stage_type == 'dual_skipconcat' and \
                        i < self.num_layers - 1:
                    b.x = torch.cat([x, b.x], dim=1)
                temp.append(b)
            batch = temp
        if cfg.gnn.l2norm:
            for b in batch:
                b.x = F.normalize(b.x, p=2, dim=-1)

        return tuple(batch)


@register_stage('dual_custom')
class GNNCustomStage(nn.Module):
    """
    Custom Stage of GNN
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (list): a list, specify the layer number of variant dim inner
    """
    def __init__(self, dim_in, dim_out, num_layers, **kwargs):
        super().__init__()

        self.num_layers = num_layers
        prev_d_out = dim_in
        layers_d_out = cfg.gnn.dim_inner
        for idx, (num_layer, d_out) in enumerate(zip(num_layers, layers_d_out)):
            layer = GNNLayer(prev_d_out, d_out, **kwargs)
            self.add_module('layer{}-{}'.format(idx, 0), layer)
            for i in range(1, num_layer):
                layer = GNNLayer(d_out, d_out, **kwargs)
                self.add_module('layer{}-{}'.format(idx, i), layer)
            prev_d_out = d_out

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            xs = [b.x for b in batch]
            batch = layer(batch)
            temp = []
            batch = list(batch)
            for idx, x in enumerate(xs):
                b = batch[idx]
                b.x = x + b.x
                temp.append(b)
            batch = temp
        if cfg.gnn.l2norm:
            for b in batch:
                b.x = F.normalize(b.x, p=2, dim=-1)

        return tuple(batch)

@register_network('CUBICASA_DUAL')
class DualGNN(GNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # if cfg.model.has_aux:
        #     self.weight_balance = nn.Parameter(torch.tensor(1, dtype=torch.float32))

    def forward(self, batch):
        res = super().forward(batch)
        return res

    def forward_test(self, batch):
        batch = self.encoder(batch)
        x_mean_0, x_var_0 = torch.mean(batch[1].x, dim=0), torch.var(batch[1].x, dim=0)
        x_mean_0 = x_mean_0.cpu().numpy()
        x_var_0 = x_var_0.cpu().numpy()

        if cfg.gnn.layers_pre_mp > 0:
            num = cfg.gnn.layers_pre_mp
            module = self.pre_mp.get_submodule('Layer_{}'.format(num - 1))
            weight_ji_pre = module.layer.forward_test(batch).cpu().numpy()
            batch = self.pre_mp(batch)

            x_mean_1, x_var_1 = torch.mean(batch[1].x, dim=0), torch.var(batch[1].x, dim=0)
            x_mean_1 = x_mean_1.cpu().numpy()
            x_var_1 = x_var_1.cpu().numpy()

        if cfg.gnn.layers_mp > 0:
            num = cfg.gnn.layers_mp # 倒数第二层看一下，不同的head有激活值？
            for i in range(num - 1):
                module = self.mp.get_submodule('layer{}'.format(i))
                weight_ji = module.layer.forward_test(batch).cpu().numpy()
                batch = module(batch)
            module = self.mp.get_submodule('layer{}'.format(num - 1))
            weight_ji = module.layer.forward_test(batch)

            x_mean_lastPre, x_var_lastPre = torch.mean(batch[1].x, dim=0), torch.var(batch[1].x, dim=0)
            x_mean_lastPre = x_mean_lastPre.cpu().numpy()
            x_var_lastPre = x_var_lastPre.cpu().numpy()

            batch = module(batch)
            x_mean_last, x_var_last = torch.mean(batch[1].x, dim=0), torch.var(batch[1].x, dim=0)
            x_mean_last = x_mean_last.cpu().numpy()
            x_var_last = x_var_last.cpu().numpy()

        _, venoGraph = self.post_mp.forward_test(batch)
        x_mean_post, x_var_post = torch.mean(batch[1].x, dim=0), torch.var(batch[1].x, dim=0)
        x_mean_post = x_mean_post.cpu().numpy()
        x_var_post = x_var_post.cpu().numpy()

        return weight_ji
