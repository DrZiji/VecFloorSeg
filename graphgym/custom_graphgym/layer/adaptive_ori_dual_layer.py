import copy

import torch
import torch.nn as nn

from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import DeepGCNLayer, MyDeepGCNLayer, GENConv, GATConv, GATv2Conv, MLP
from torch_geometric.graphgym.contrib.layer.CUBI_2_layer import \
    GATv2_EdgeAttn, GATv2_MLP, \
    GATv3_MLP, GATv3_PanAttn_0, GATv3_PanAttn_1, GATv3_PanAttn_3, GATv3_PanAttn_4, \
    GATv3_Identity


NodeAttnModuleName = [
    'Graph Attn v2', 'GATv2+Edge Attn', 'GATv2+MLP', 'GATv3+MLP',
    'GATv3_PanAttn_0', 'GATv3_PanAttn_1', 'GATv3_PanAttn_3', 'GATv3_dilated',
    'Graph Attn', 'GATv3_PanAttn_4', 'GATv3_Identity'

]
EdgeAttnModuleName = [
    'Graph Attn v2', 'GATv2+MLP', 'GATv3+MLP', 'GATv2+MLP+Dual', 'GATv3_PanAttn_0',
    'GATv3_PanAttn_1', 'Graph Attn', 'GATv3_PanAttn_4', 'GATv3_Identity'
]



class DeepGCN_Node(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        ConvType = NodeConvDict[kwargs['node_layer_type']]

        if kwargs['node_layer_type'] in NodeAttnModuleName:
            dim_in, dim_out = \
                layer_config.dim_in, layer_config.dim_out // kwargs['__node_heads']
        else:
            dim_in, dim_out = layer_config.dim_in, layer_config.dim_out

        ConvArgs = {}
        PREFIX_NUM = kwargs['prefix_num']
        for k, v in kwargs.items():
            if '__node_' in k and k != 'node_layer_type':
                ConvArgs[k[PREFIX_NUM:]] = v

        conv = ConvType(
            dim_in, dim_out,
            **ConvArgs
        )
        if kwargs['has_norm_act']:
            norm = nn.LayerNorm(dim_in, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = MyDeepGCNLayer(
                conv, norm, act, block=layer_config.layer_order, dropout=0.1
            )
            self.model = layer
        else:
            self.model = conv

        self.edgeMLP = nn.Sequential(
            nn.LayerNorm(layer_config.dim_out * 2),
            nn.ReLU(inplace=True),
            nn.Linear(layer_config.dim_out * 2, layer_config.dim_out)
        )

        if kwargs.__contains__('intermediate_output'):
            self.inter = True
        else:
            self.inter = False
        self.inter_output = None

    def forward(self, graph):
        x, pair = self.model(
            graph.x, graph.edge_index, graph.edge_attr,
            return_attention_weights=True
        )
        x_src, x_dst = x[graph.edge_index[0]], x[graph.edge_index[1]]
        eAttr = self.edgeMLP(torch.cat([x_src, x_dst], dim=-1))
        graph.edge_attr = graph.edge_attr + eAttr
        graph.x = x

        if self.inter:
            self.inter_output = pair[1]

        return graph

    def reset_param(self):
        if self.inter_output is not None:
            self.inter_output = None

    def forward_test(self, graph):
        edge_attr = graph.edge_attr
        x, pair = self.model(
            graph.x, graph.edge_index, edge_attr,
            return_attention_weights=True
        )

        return pair[1]


class DeepGCN_Node_Dual(DeepGCN_Node):
    def __init__(self, layer_config, **kwargs):
        super(DeepGCN_Node_Dual, self).__init__(layer_config, **kwargs)


    def forward(self, graph):
        edge_attr = graph.dual_attr
        x, pair = self.model(
            graph.x, graph.edge_index, edge_attr,
            return_attention_weights=True
        )
        x_src, x_dst = x[graph.edge_index[0]], x[graph.edge_index[1]]
        eAttr = self.edgeMLP(torch.cat([x_src, x_dst], dim=-1))
        graph.edge_attr = graph.edge_attr + eAttr
        graph.x = x

        if self.inter:
            self.inter_output = pair[1]

        return graph

    def forward_test(self, graph):
        edge_attr = graph.dual_attr
        x, pair = self.model(
            graph.x, graph.edge_index, edge_attr,
            return_attention_weights=True
        )

        return pair[1]


class DeepGCN_Edge(nn.Module):
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        ConvType = EdgeConvDict[kwargs['edge_layer_type']]

        if kwargs['edge_layer_type'] in EdgeAttnModuleName:
            dim_in, dim_out = \
                layer_config.dim_in, layer_config.dim_out // kwargs['__edge_heads']
        else:
            dim_in, dim_out = layer_config.dim_in, layer_config.dim_out

        ConvArgs = {}
        PREFIX_NUM = kwargs['prefix_num']
        for k, v in kwargs.items():
            if '__edge_' in k and k != 'edge_layer_type':
                ConvArgs[k[PREFIX_NUM:]] = v

        conv = ConvType(
            dim_in, dim_out,
            **ConvArgs
        )
        if kwargs['has_norm_act']:
            norm = nn.LayerNorm(dim_in, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(
                conv, norm, act, block=layer_config.layer_order, dropout=0.1
            )
            self.model = layer
        else:
            self.model = conv

        # self.edgeMLP = MLP(channel_list=[layer_config.dim_out * 2, layer_config.dim_out],)
        self.edgeMLP = nn.Sequential(
            nn.LayerNorm(layer_config.dim_out * 2),
            nn.ReLU(inplace=True),
            nn.Linear(layer_config.dim_out * 2, layer_config.dim_out)
        )

    def forward(self, graph):
        x = self.model(graph.x, graph.edge_index, graph.edge_attr)
        x_src, x_dst = x[graph.edge_index[0]], x[graph.edge_index[1]]
        eAttr = self.edgeMLP(torch.cat([x_src, x_dst], dim=-1))
        graph.edge_attr = graph.edge_attr + eAttr
        graph.x = x
        return graph



class DeepGCN_Edge_Dual(DeepGCN_Edge):
    # 相比于DeepGCN_Edge，本类使用了来自另一个分支上的边特征参与更新。
    # 即，使用了venoGraph的边特征
    def __init__(self, layer_config, **kwargs):
        super().__init__(layer_config, **kwargs)


    def forward(self, graph):
        edge_attr = graph.dual_attr
        x = self.model(graph.x, graph.edge_index, edge_attr)
        x_src, x_dst = x[graph.edge_index[0]], x[graph.edge_index[1]]
        eAttr = self.edgeMLP(torch.cat([x_src, x_dst], dim=-1))
        graph.edge_attr = graph.edge_attr + eAttr
        graph.x = x
        return graph



class PlaceHolder(nn.Module):
    # 在edge/node stage，不对边特征进行更新
    def __init__(self, layer_config, **kwargs):
        super().__init__()

    def forward(self, graph):
        return graph



class Node2EdgePlaceHolder:
    def __init__(self):
        pass

    def __call__(self, oriGraph, venoGraph):
        return oriGraph



class Node2EdgeNaiveReplace:
    # 在oriGraph中添加边的新属性，对偶边特征（由venoGraph中的边顶点特征concat得到）
    def __init__(self):
        pass

    def __call__(self, oriGraph, venoGraph):
        edge_attr = venoGraph.edge_attr
        ones_padding = torch.ones(
            (1, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, ones_padding], dim=0)
        oemap_idx = oriGraph.edge_dual_idx.squeeze()
        oriGraph.dual_attr = edge_attr[oemap_idx]
        return oriGraph


class Edge2NodePlaceHolder:
    def __init__(self):
        pass

    def __call__(self, oriGraph, venoGraph):
        return venoGraph



class Edge2NodeNavieReplace:
    # 将original graph中的边特征替换到veno graph上
    def __init__(self):
        pass

    def __call__(self, oriGraph, venoGraph):
        vemap_idx = venoGraph.edge_dual_idx.squeeze()
        venoGraph.dual_attr = oriGraph.edge_attr[vemap_idx]
        return venoGraph


NodeConvDict = {
    'General Conv': GENConv,
    'Graph Attn': GATConv,
    'Graph Attn v2': GATv2Conv,
    'GATv2+Edge Attn': GATv2_EdgeAttn,
    'GATv2+MLP': GATv2_MLP,
    'GATv3+MLP': GATv3_MLP,
    'GATv3_PanAttn_0': GATv3_PanAttn_0,
    'GATv3_PanAttn_1': GATv3_PanAttn_1,
    'GATv3_PanAttn_3': GATv3_PanAttn_3,
    'GATv3_PanAttn_4': GATv3_PanAttn_4,
    'GATv3_Identity': GATv3_Identity
}

EdgeConvDict = {
    'GATv2+MLP': GATv2_MLP,
    'GATv3+MLP': GATv3_MLP,
    'Graph Attn v2': GATv2Conv,
    'GATv3_PanAttn_0': GATv3_PanAttn_0,
    'GATv3_PanAttn_1': GATv3_PanAttn_1,
    'Graph Attn': GATConv,
    'GATv3_PanAttn_4': GATv3_PanAttn_4,
    'GATv3_Identity': GATv3_Identity
}

NodeStageDict = {
    'Nothing': None,
    'placeholder': PlaceHolder,
    'DeepGCN': DeepGCN_Node,
    'DeepGCN_Dual': DeepGCN_Node_Dual,
}

Node2EdgeDict = {
    'placeholder': Node2EdgePlaceHolder,
    'naive replacement': Node2EdgeNaiveReplace
}

EdgeStageDict = {
    'Nothing': None,
    'placeholder': PlaceHolder,
    'DeepGCN': DeepGCN_Edge,
    'DeepGCN_Dual': DeepGCN_Edge_Dual
}

Edge2NodeDict = {
    'placeholder': Edge2NodePlaceHolder,
    'naive replacement': Edge2NodeNavieReplace,
}



@register_layer("NodeEdgeLayer")
class NodeEdgeLayer(nn.Module):
    """
    每一层分为两个stage, node stage和edge stage
    node stage在veno图上运算，更新图中节点的特征
    edge stage在original图上运算，更新图中边的特征

    两个stage需要进行特征的映射，从veno -> original是节点特征的映射
    从original -> veno是边特征的映射

    注意可扩展性,如果node_stage没有的话,则退化成edge network
    如果edge stage没有的话,则退化成node network
    """
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        nodeStage = NodeStageDict[kwargs['nodestage']]
        edgeStage = EdgeStageDict[kwargs['edgestage']]
        if nodeStage is not None:
            self.nodeStage = nodeStage(layer_config, **kwargs)
        else:
            self.nodeStage = None

        if edgeStage is not None:
            self.edgeStage = edgeStage(layer_config, **kwargs)
        else:
            self.edgeStage = None

        node2edgeName = kwargs['node2edge']
        edge2nodeName = kwargs['edge2node']
        if self.nodeStage is not None: # veno图更新了node feature后，将特征并入ori图
            self.node2edge = Node2EdgeDict[node2edgeName]()
        if self.edgeStage is not None: # 原图更新了edge feature后，将特征并入veno图
            self.edge2node = Edge2NodeDict[edge2nodeName]()

    def forward(self, batch):
        fplanGraph, venoGraph = batch

        if self.nodeStage is not None:
            venoGraph = self.nodeStage(venoGraph)
            fplanGraph = self.node2edge(fplanGraph, venoGraph)

        if self.edgeStage is not None:
            fplanGraph = self.edgeStage(fplanGraph)
            venoGraph = self.edge2node(fplanGraph, venoGraph)

        return fplanGraph, venoGraph

    def forward_test(self, batch):
        temp_batch = copy.deepcopy(batch)
        fplanGraph, venoGraph = temp_batch

        if self.nodeStage is not None:
            return self.nodeStage.forward_test(venoGraph)


@register_layer('NodeLayer')
class NodeLayer(nn.Module):
    # 在prestage中将四个dialation的特征concat起来；
    # dilated gcn conv使用GAT Ours，其中的边特征使用两个节点路径上的所有节点的特征平均
    def __init__(self, layer_config, **kwargs):
        super().__init__()
        dim_in, dim_out = \
            layer_config.dim_in, layer_config.dim_out // 16
        self.conv1 = GATv3_PanAttn_0(
            dim_in, dim_out, heads=16, plain_last=False, num_layers=1, fill_value='eye',
            dropout=0.1
        )
        self.conv2 = GATv3_PanAttn_1(
            dim_in, dim_out, heads=16, edge_heads=2, plain_last=False, num_layers=1,
            fill_value='eye', dropout=0.1
        )
        self.conv3 = GATv3_PanAttn_1(
            dim_in, dim_out, heads=16, edge_heads=2, plain_last=False, num_layers=1,
            fill_value='eye', dropout=0.1
        )
        # self.conv4 = GATv3_PanAttn_1(
        #     dim_in, dim_out, heads=16, edge_heads=4, plain_last=False, num_layers=1,
        #     fill_value='eye', dropout=0.1
        # )

        self.fuse = nn.Linear(dim_out * 16 * 3, dim_out * 16)
        self.edgeMLP = nn.Sequential(
            nn.LayerNorm(layer_config.dim_out * 2),
            nn.ReLU(inplace=True),
            nn.Linear(layer_config.dim_out * 2, layer_config.dim_out)
        )
        self.node2edge = Node2EdgeNaiveReplace()

    def edgeIndex2Adj(self, edge_index):
        """
        规定adj的维度0表示edge_index的维度0，维度1表示edge_index的维度1
        :param edge_index: (tensor)
        :return:
        """
        minVIdx, maxVIdx = 0, torch.max(edge_index) - torch.min(edge_index)
        adj = torch.zeros([maxVIdx + 1, maxVIdx + 1], dtype=torch.int).view(-1)
        adj_coord = edge_index[0] * (maxVIdx + 1) + edge_index[1]

        adj[adj_coord] = 1  # 由于经过了数据集的transform，所以得到的是有向图。
        adj = adj.view(maxVIdx + 1, maxVIdx + 1)
        return adj # 对角线为0

    def removeRepeated(self, adj_1, *adj_2):
        # 跟多个不同阶数的邻接矩阵做按位与操作
        adj_1 = adj_1.to(torch.bool)
        diagonal = torch.eye(adj_1.shape[0]).to(adj_1.device)
        diagonal = diagonal.to(torch.bool)

        for adj in adj_2:
            adj = adj.to(torch.bool)
            adj_1 = torch.logical_and(adj_1, torch.logical_not(adj))

        adj_1 = torch.logical_and(adj_1, torch.logical_not(diagonal))
        adj_1 = adj_1.to(torch.float32)
        return adj_1

    def pathwayMeanFeat(self, v_feat, adj_mat, dilation=4):
        v_feat_new = v_feat.unsqueeze(0)
        # 根据adj_mat的第一个维度，将v_feat填入到adj_mat中
        adj_mat = adj_mat.to(v_feat.device).to(torch.float)
        adj_feat = adj_mat.unsqueeze(-1) * v_feat_new

        adjs = [adj_mat]  # 存储不同阶数的邻接矩阵(从1阶开始)
        numV_adjs = [torch.zeros_like(adj_mat)]  # 在不同阶数的邻接矩阵上，存储两个节点之间经过的节点数量()
        feat_adjs = [adj_feat]  # 在不同阶数的邻接矩阵上，存储起始节点到达目标节点过程中，经过的所有节点的特征和

        adj_feat_pres = adj_feat # n,n,c
        numV_mat_pres = adj_mat
        for i in range(2, dilation + 1):
            # 右乘邻接矩阵，
            adj_feat_next = torch.einsum("xyc,yzc->xzc", adj_feat_pres, adj_mat.unsqueeze(-1))
            numV_mat_next = torch.mm(numV_mat_pres, adj_mat) # 路径上的点的数量

            # 并删掉重复和低阶邻接矩阵重复的位置(通过adj_mat_next操作得到)
            adj_mat_next = self.removeRepeated(numV_mat_next, *tuple(adjs))
            adj_feat_next = adj_feat_next * adj_mat_next.unsqueeze(-1)
            numV_mat_next = numV_mat_next * adj_mat_next

            # 为下一个dilation准备邻接矩阵和特征矩阵
            adj_feat_pres = adj_feat_next + adj_mat_next.unsqueeze(-1) * v_feat_new
            numV_mat_pres = numV_mat_next + adj_mat_next

            # 将这一阶的结果存到列表中
            adjs.append(adj_mat_next)
            numV_adjs.append(numV_mat_next)
            feat_adjs.append(adj_feat_next)

        return adjs, numV_adjs, feat_adjs

    def Adj2EdgeIndex(self, adjs, adj_feats, adj_numVs):

        edge_indexes, edge_attrs = [], []
        for adj, feat, numV in zip(adjs[1:], adj_feats[1:], adj_numVs[1:]):
            h, w = adj.shape
            coord = torch.arange(0, h * w, step=1)
            feat = feat.contiguous().view(h * w, -1)
            numV = numV.contiguous().view(h * w, -1)

            mask = adj.view(-1) > 0
            index_flatten = coord[mask]
            h_coord = torch.div(index_flatten, w, rounding_mode='trunc')
            w_coord = index_flatten - h_coord * w
            edge_index = torch.cat([h_coord.unsqueeze(0), w_coord.unsqueeze(0)], dim=0)
            edge_attr = feat[mask] / (numV[mask] + 1e-7)

            edge_indexes.append(edge_index.to(device=adj.device, dtype=torch.long))
            edge_attrs.append(edge_attr)

        return edge_indexes, edge_attrs

    def forward(self, batch):
        # 思路：利用邻接矩阵的左乘和右乘，将通路上的特征求和
        # 具体步骤：
        # (1)根据edge index建立一个邻接矩阵，根据邻接矩阵建立对应的特征矩阵。假设(x,y)的值为1,则在
        # (x,y)上加入y点的特征值
        # (2)在特征矩阵上右乘邻接矩阵,假设dilation为2，则右乘一次邻接矩阵。
        #   这样做的含义是：吧通路上的端点特征，通过矩阵相乘的方式求和
        # (3)假设dilation为3，首先在(2)的结果上，再加一个1阶邻接矩阵生成的特征矩阵，然后仿照(2)再操作一次
        # 最后要求路径上特征的平均值，
        # 根据邻接矩阵，维护一个路径上有多少点的个数的矩阵，也是通过邻接矩阵来算
        # 注意细节:如果在2阶邻接矩阵上，则不要考虑3阶邻接矩阵

        # 存在问题：加入通过某个节点延伸出去多条路径，则这个节点的值会被重复计算,不过感觉影响不会太大，试试吧
        fplanGraph, venoGraph = batch
        x1 = self.conv1(
            venoGraph.x, venoGraph.edge_index, venoGraph.dual_attr
        )

        x_feat = venoGraph.x.clone()
        Adj_mat = self.edgeIndex2Adj(venoGraph.edge_index)
        adjs, numV_adjs, feat_adjs = self.pathwayMeanFeat(x_feat, Adj_mat, dilation=4)
        edge_idxes, edge_attrs = self.Adj2EdgeIndex(adjs, feat_adjs, numV_adjs)

        x2 = self.conv2(venoGraph.x, edge_idxes[0], edge_attrs[0])
        x3 = self.conv3(venoGraph.x, edge_idxes[2], edge_attrs[2])
        # x4 = self.conv4(venoGraph.x, venoGraph.edge_index_d8, None)

        x = self.fuse(torch.cat([x1, x2, x3], dim=1))
        x_src, x_dst = x[venoGraph.edge_index[0]], x[venoGraph.edge_index[1]]
        eAttr = self.edgeMLP(torch.cat([x_src, x_dst], dim=-1))
        venoGraph.edge_attr = venoGraph.edge_attr + eAttr
        venoGraph.x = x

        fplanGraph = self.node2edge(fplanGraph, venoGraph)

        return fplanGraph, venoGraph

    def forward_test(self, batch):
        fplanGraph, venoGraph = batch
        edge_attr = venoGraph.dual_attr
        x, pair = self.conv1(
            venoGraph.x, venoGraph.edge_index, edge_attr,
            return_attention_weights=True
        )

        return pair[1]


@register_layer('NodeLayer1')
class NodeLayer1(nn.Module):
    # 在NodeLayer1中不应用edge stage，因为这一部分没有residual connection
    # 所以不能保存完整的edge extra信息。
    # 在prestage中，将四个dilation的特征全部合并起来
    # dilation使用GATv3_MLP
    def __init__(self, layer_config, **kwargs):
        super(NodeLayer1, self).__init__()

        dim_in, dim_out = \
            layer_config.dim_in, layer_config.dim_out // 16
        self.conv1 = GATv3_PanAttn_0(
            dim_in, dim_out, heads=16, plain_last=False, num_layers=1, fill_value='eye', dropout=0.1)
        self.conv2 = GATv3_MLP(dim_in, dim_out, heads=16, plain_last=False, num_layers=1, dropout=0.1)
        self.conv3 = GATv3_MLP(dim_in, dim_out, heads=16, plain_last=False, num_layers=1, dropout=0.1)
        self.conv4 = GATv3_MLP(dim_in, dim_out, heads=16, plain_last=False, num_layers=1, dropout=0.1)

        self.fuse = nn.Linear(dim_out * 16 * 4, dim_out * 16)
        self.edgeMLP = nn.Sequential(
            nn.LayerNorm(layer_config.dim_out * 2),
            nn.ReLU(inplace=True),
            nn.Linear(layer_config.dim_out * 2, layer_config.dim_out)
        )
        self.node2edge = Node2EdgeNaiveReplace()

    def forward(self, batch):
        fplanGraph, venoGraph = batch
        x1 = self.conv1(
            venoGraph.x, venoGraph.edge_index, venoGraph.dual_attr
        )
        x2 = self.conv2(venoGraph.x, venoGraph.edge_index_d2, None)
        x3 = self.conv3(venoGraph.x, venoGraph.edge_index_d4, None)
        x4 = self.conv4(venoGraph.x, venoGraph.edge_index_d8, None)

        x = self.fuse(torch.cat([x1, x2, x3, x4], dim=1))
        x_src, x_dst = x[venoGraph.edge_index[0]], x[venoGraph.edge_index[1]]
        eAttr = self.edgeMLP(torch.cat([x_src, x_dst], dim=-1))
        venoGraph.edge_attr = venoGraph.edge_attr + eAttr
        venoGraph.x = x

        fplanGraph = self.node2edge(fplanGraph, venoGraph)

        return fplanGraph, venoGraph

    def forward_test(self, batch):
        fplanGraph, venoGraph = batch
        edge_attr = venoGraph.dual_attr
        x, pair = self.conv1(
            venoGraph.x, venoGraph.edge_index, edge_attr,
            return_attention_weights=True
        )

        return pair[1]


if __name__ == "__main__":
    class LayerConfig:
        dim_in = 256
        dim_out = 16

    a = NodeLayer(LayerConfig())
    edge_index = torch.tensor(
        [[0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
         [1, 0, 2, 3, 1, 4, 1, 4, 2, 3]]
        # [[0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7],
        #  [1, 0, 2, 3, 1, 4, 1, 4, 2, 3, 5, 6, 4, 7, 4, 7, 5, 6]]
    )
    adj = a.edgeIndex2Adj(edge_index)
    vFeat = torch.tensor(
        [[0.5, 0.5],
        [1, 1],
         [2, 2],
         [3, 3],
         [4, 4],
         ]
        #          [5, 5],
        #          [6, 6],
        #          [7, 7]
    )
    adjs, adjfeats, adjNums = a.pathwayMeanFeat(vFeat, adj_mat=adj)
    print()
