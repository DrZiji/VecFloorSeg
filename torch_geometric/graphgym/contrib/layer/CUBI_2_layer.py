import torch
import torch.nn  as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor
from torch_scatter import scatter, scatter_softmax
from torch_sparse import SparseTensor, set_diag
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import GENConv, GATv2Conv, MLP, AGNNConv
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class EdgeWeight(MessagePassing):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.edge_embedding = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=1),
            nn.BatchNorm1d(1), # 这里的BatchNorm可能要解决一下
            nn.Sigmoid()
        )

    def edge_weight(self, edge_index, edge_attr):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        # 第零步，对edge_attr做变换
        edge_struc = self.edge_embedding(edge_attr)

        # 第一步，将edge_attr聚集到节点上,并统计每一个节点结合了几个edge_attr
        numEdges_on_node = torch.ones_like(edge_index[i]).unsqueeze(-1)
        numEdges_on_node = scatter(numEdges_on_node, edge_index[i], dim=-2)
        edges_on_node = scatter(edge_struc, edge_index[i], dim=-2)

        # 第二步，将edges_on_node __lift__()到edge_index的维度，
        # 分别代表了edge source和target端的节点所保存的edge_attr
        edges_on_tgt = self.__lift__(edges_on_node, edge_index, dim=j)
        numEdges_on_tgt = self.__lift__(numEdges_on_node, edge_index, dim=j)
        edges_on_src = self.__lift__(edges_on_node, edge_index, dim=i)
        numEdges_on_src = self.__lift__(numEdges_on_node, edge_index, dim=i)

        # 第三步，减去source和target节点相连接的这条edge_attr
        edges_on_tgt, edges_on_src = edges_on_tgt - edge_struc, edges_on_src - edge_struc

        # 第四步，将node_tgt和node_src相加，作为结构化特征知道两个节点attention
        # weight
        weight_ji = (edges_on_tgt + edges_on_src) / \
                    (numEdges_on_tgt + numEdges_on_src - 2)

        return weight_ji


class GATv2_EdgeAttn(GATv2Conv):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super(GATv2_EdgeAttn, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            **kwargs
        )
        self.ew = EdgeWeight(in_channels=in_channels)
        self.mlp = MLP(
            in_channels=in_channels, hidden_channels=in_channels * 2,
            out_channels=in_channels, num_layers=2, act_first=False,
            plain_last=True, dropout=dropout
        )

    def message(self, x_j, x_i, edge_attr, index, edge_index, ptr, size_i):
        x = x_i + x_j

        if edge_attr is not None:
            # if edge_attr.dim() == 1:
            #     edge_attr = edge_attr.view(-1, 1)
            # assert self.lin_edge is not None
            # edge_attr = self.lin_edge(edge_attr)
            # edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            # x += edge_attr

            # 针对GAT可能有自环，先将自环拆掉计算weight_ij，然后将自环上的weight_ii补成0
            if self.add_self_loops:
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            weight_ij = self.ew.edge_weight(edge_index, edge_attr)
            if self.add_self_loops:
                edge_index, weight_ij = add_self_loops(edge_index, weight_ij, fill_value=0.)

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = alpha + weight_ij # 模仿在graphormer中的相加方式
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * (alpha.unsqueeze(-1))

    def forward(self, x, edge_index,
                edge_attr=None,
                return_attention_weights=None):

        # 注意: 这里没有考虑到，当return_attention_weights为True的时候
        # 会返回attention权重，因此这里严谨地说，是存在问题的。
        out = super().forward(x, edge_index, edge_attr, return_attention_weights)
        if isinstance(return_attention_weights, bool):
            return self.mlp(out[0]), out[1]
        else:
            return self.mlp(out)

    def forward_test(self):
        pass


class GATv2_MLP(GATv2Conv):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        norm='batch',
        plain_last=True,
        num_layers=2,
        **kwargs,
    ):
        super(GATv2_MLP, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            **kwargs
        )
        self.mlp = MLP(
            in_channels=in_channels, hidden_channels=in_channels * 2,
            out_channels=in_channels, num_layers=num_layers, act_first=False,
            plain_last=plain_last, batch_norm=norm, dropout=dropout
        )

    def forward(self, x, edge_index,
                edge_attr=None,
                return_attention_weights=None):
        out = super().forward(x, edge_index, edge_attr, return_attention_weights)
        if return_attention_weights is not None:
            return self.mlp(out[0]), out[1]
        else:
            return self.mlp(out)


class GATv3_MLP(GATv2_MLP):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        norm='batch',
        plain_last=True,
        num_layers=2,
        **kwargs,
    ):
        super(GATv3_MLP, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )

        self.sqrt_dk = torch.sqrt(
            torch.tensor(out_channels, dtype=torch.float)
        )

        if isinstance(in_channels, int):
            if share_weights:
                self.lin_v = self.lin_l
            else:
                self.lin_v = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            if share_weights:
                self.lin_v = self.lin_l
            else:
                self.lin_v = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        self.lin_v.reset_parameters()

    def forward(self, x, edge_index,
                edge_attr=None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
                xv_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
                xv_r = self.lin_v(x).view(-1, H, C)
        else:
            x_l, x_r, xv_r = x[0], x[1], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                xv_r = self.lin_v(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        assert xv_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        xv_r = self.__lift__(xv_r, edge_index, j)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             x_v=xv_r, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return self.mlp(out), (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return self.mlp(out), edge_index.set_value(alpha, layout='coo')
        else:
            return self.mlp(out)



    def message(self, x_j, x_i, x_v, edge_attr,
                index, ptr,
                size_i):
        x_i = x_i.unsqueeze(-2)
        x_j = x_j.unsqueeze(-1)
        weights = torch.einsum('nhxy,nhyz->nhxz', x_i, x_j)  # num_edge, H, 1, 1
        # weights = torch.sum(weights * self.att, dim=-1) # num_edge,H
        weights = weights.squeeze() / self.sqrt_dk

        weights = softmax(weights, index, ptr, size_i)
        self._alpha = weights
        alpha = F.dropout(weights, p=self.dropout, training=self.training)
        return x_v * alpha.unsqueeze(-1)


class GATv3_PanAttn_0(GATv3_MLP):
    # 在venoGraph的节点相似度比较中，加入一个由edge attr生成的
    # 非对称方阵，用来调制节点的相似度比较，从而引入了新的信息。
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        norm='batch',
        plain_last=True,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )

        # 对边的scale处理；
        self.pre_embeds = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        edgeEmbeds = nn.ModuleList()
        for i in range(heads): # 最好将heads设置为8或16，不然参数量实在太大了。
            edgeEmbeds.append(nn.Linear(in_channels, out_channels * out_channels))
        self.eEmbeds = edgeEmbeds

    def forward(self, x, edge_index,
                edge_attr=None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
                xv_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
                xv_r = self.lin_v(x).view(-1, H, C)
        else:
            x_l, x_r, xv_r = x[0], x[1], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                xv_r = self.lin_v(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        assert xv_r is not None

        edge_attr = self.pre_embeds(edge_attr)
        edge_attrs = []
        for embed in self.eEmbeds:
            e_attr = embed(edge_attr)
            edge_attrs.append(e_attr)
        edge_attr = torch.cat(edge_attrs, dim=-1)

        assert self.fill_value == 'eye'
        fill_value = torch.eye(n=self.out_channels, m=self.out_channels).unsqueeze(0)
        fill_value = fill_value.repeat(self.heads, 1, 1).view(-1)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        xv_r = self.__lift__(xv_r, edge_index, j)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             x_v=xv_r, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return self.mlp(out), (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return self.mlp(out), edge_index.set_value(alpha, layout='coo')
        else:
            return self.mlp(out)

    def message(self, x_j, x_i, x_v, edge_attr,
                index, ptr,
                size_i):
        H, C = self.heads, self.out_channels
        x_i, x_j = x_i.unsqueeze(-2), x_j.unsqueeze(-1)
        edge_attr = edge_attr.view(-1, H, C, C)
        weights = torch.einsum(
            'nhxy,nhyz->nhxz', x_i, edge_attr)
        weights = torch.einsum(
            'nhxy,nhyz->nhxz', weights, x_j
        )
        weights = weights.squeeze() / self.sqrt_dk

        weights = softmax(weights, index, ptr, size_i)
        self._alpha = weights
        alpha = F.dropout(weights, p=self.dropout, training=self.training)
        return x_v * alpha.unsqueeze(-1)

    def forward_test(self, x, edge_index, edge_attr):
        feat_out, attn = \
            self.forward(x, edge_index, edge_attr, return_attention_weights=True)
        return attn[1]


class GATv3_PanAttn_1(GATv3_MLP):
    # 在PanAttn_0的基础上，将非堆成矩阵进行因式分解，降低占用参数量
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        edge_heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        norm='batch',
        plain_last=True,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )

        self.pre_embeds = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        edgeEmbeds = nn.ModuleList()
        for i in range(edge_heads): # 最好将heads设置为8或16，不然参数量实在太大了。
            edgeEmbeds.append(nn.Linear(in_channels, out_channels * out_channels))
        self.eEmbeds = edgeEmbeds

    def forward(self, x, edge_index,
                edge_attr=None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
                xv_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
                xv_r = self.lin_v(x).view(-1, H, C)
        else:
            x_l, x_r, xv_r = x[0], x[1], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                xv_r = self.lin_v(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        assert xv_r is not None

        edge_attr = self.pre_embeds(edge_attr)
        edge_attrs = []
        for embed in self.eEmbeds:
            e_attr = embed(edge_attr)
            edge_attrs.append(e_attr)
        edge_attr = torch.cat(edge_attrs, dim=-1)

        expand_size = self.heads // len(self.eEmbeds)
        eshape1 = edge_attr.shape[1]
        edge_attr = edge_attr.unsqueeze(1).expand(-1, expand_size, -1)
        edge_attr = edge_attr.contiguous().view(-1, eshape1 * expand_size)

        assert self.fill_value == 'eye'
        fill_value = torch.eye(n=self.out_channels, m=self.out_channels).unsqueeze(0)
        fill_value = fill_value.repeat(self.heads, 1, 1).view(-1)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)
        xv_r = self.__lift__(xv_r, edge_index, j)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             x_v=xv_r, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return self.mlp(out), (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return self.mlp(out), edge_index.set_value(alpha, layout='coo')
        else:
            return self.mlp(out)


class GATv3_PanAttn_2(GATv3_PanAttn_0):
    # 模仿 K-Means Mask Transformer，对 attention weights做direct supervision
    # 方式，由于attention weight生成的是n, head，定义一个叫做weight aggr的linear层
    # 该层将n, head聚合为final attn w: n,1，该值进行sigmoid，再和v(n, h, c相乘)
    # final attn w传到类外，利用Dice loss约束。
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        rank: int = 8,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        norm='batch',
        plain_last=True,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )
        self.weight_aggr = nn.Linear(in_features=heads, out_features=1)
        self.temperature = 2

    def message(self, x_j, x_i, x_v, edge_attr,
                index, ptr,
                size_i):
        H, C = self.heads, self.out_channels
        x_i, x_j = x_i.unsqueeze(-2), x_j.unsqueeze(-1)
        edge_attr = edge_attr.view(-1, H, C, C)
        weights = torch.einsum(
            'nhxy,nhyz->nhxz', x_i, edge_attr)
        weights = torch.einsum(
            'nhxy,nhyz->nhxz', weights, x_j
        )
        weights = weights.squeeze() / self.sqrt_dk
        weights = self.weight_aggr(weights)
        # weights = self.weight_act(weights)
        self._alpha = weights

        # 正常来说，为了特征的聚合不出现scale的问题，需要对weight做一次归一化，
        # 但是norm层是否可以做到这一点？在res+的结构中norm没有和conv层挨着，还是
        # 用softmax归一化吧，但是注意设置temperature
        weights = softmax(weights / self.temperature, index, ptr, size_i)
        alpha = F.dropout(weights, p=self.dropout, training=self.training)
        return x_v * alpha.unsqueeze(-1)


class GATv3_PanAttn_3(GATv3_PanAttn_0):
    # 对于边特征的处理，将其中的ReLU去掉
    def __init__(
            self,
            in_channels,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim=None,
            fill_value='mean',
            bias: bool = True,
            share_weights: bool = False,
            norm='batch',
            plain_last=True,
            num_layers=2,
            **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )
        self.pre_embeds = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Dropout(dropout)
        )


class GATv3_PanAttn_4(GATv3_PanAttn_1):
    # 对于边特征的处理，将其中的ReLU去掉
    def __init__(
            self,
            in_channels,
            out_channels: int,
            heads: int = 1,
            edge_heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim=None,
            fill_value='mean',
            bias: bool = True,
            share_weights: bool = False,
            norm='batch',
            plain_last=True,
            num_layers=2,
            **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, edge_heads=edge_heads,
            concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )
        self.mlp = nn.Identity()
        self.sqrt_dk = self.sqrt_dk / 4


class GATv3_Identity(GATv3_MLP):
    # 将GATv3_MLP中的MLP变成Identity
    def __init__(
        self,
        in_channels,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim=None,
        fill_value='mean',
        bias: bool = True,
        share_weights: bool = False,
        norm='batch',
        plain_last=True,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads, concat=concat, negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops, edge_dim=edge_dim,
            fill_value=fill_value, bias=bias, share_weights=share_weights,
            norm=norm, plain_last=plain_last, num_layers=num_layers,
            **kwargs
        )

        self.mlp = nn.Identity()
