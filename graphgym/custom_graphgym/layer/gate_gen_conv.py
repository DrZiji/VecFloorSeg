import torch

from torch import Tensor
from torch_sparse import SparseTensor
from typing import List, Optional, Union
from torch_geometric.nn.conv.gen_conv import GENConv
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_scatter import scatter, scatter_sum, scatter_max
from torch_scatter.utils import broadcast
from torch_sparse import SparseTensor


def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max(
        src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter_sum(
        recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)
    normalizing_constants += 1e-7

    return recentered_scores_exp.div(normalizing_constants)


class GateGenConv(GENConv):
    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = 'softmax', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False, msg_norm: bool = False,
                 learn_msg_scale: bool = False, norm: str = 'batch',
                 num_layers: int = 2, eps: float = 1e-7, **kwargs):

        super().__init__(in_channels, out_channels,
                 aggr, t, learn_t,
                 p, learn_p, msg_norm,
                 learn_msg_scale, norm,
                 num_layers, eps, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_type: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr,
            edge_type=edge_type, size=size
        )

        if self.msg_norm is not None:
            out = self.msg_norm(x[0], out)

        x_r = x[1]
        if x_r is not None:
            out += x_r

        return self.mlp(out)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  edge_type, dim_size: Optional[int] = None) -> Tensor:
        # 对inputs进行处理,相当于加一个mask,inputs和edge_type的shape一致
        # inputs是按照edge_index中的source扩充的;
        # index等同于edge_index中target,

        # 注意,这里不能设置为负无穷,会导致梯度反传出现问题,可能是梯度反传时出现0值
        # inputsGate = torch.where(edge_type == 0, -10000., edge_type.to(torch.double))
        # inputsGate = torch.where(inputsGate == 1, 0., inputsGate)
        inputsGate = (edge_type - 1) * 10000
        inputsGate = inputsGate.to(torch.float)

        if self.aggr == 'softmax':
            inputWeights = inputsGate + inputs
            out = scatter_softmax(inputWeights * self.t, index, dim=self.node_dim)
            targets = scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')
            return targets

        elif self.aggr == 'softmax_sg':
            inputWeights = inputsGate * inputs
            out = scatter_softmax(inputWeights * self.t, index,
                                  dim=self.node_dim).detach()
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        elif self.aggr == 'power':
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1 / self.p)

        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)
