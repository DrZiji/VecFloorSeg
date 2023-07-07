import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.graphgym.register import (
    register_node_encoder,
)
from .node_edge_encoder import ImgEncoderDict, DualNodeEncoder
from torch_geometric.graphgym.config import cfg


class MyGCNConv(GCNConv):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         improved=improved, cached=cached,
                         add_self_loops=add_self_loops, normalize=normalize,
                         bias=bias, **kwargs)
        self.lin = nn.Identity()



@register_node_encoder('merge_node')
class MergeNodeEncoder(DualNodeEncoder):
    def __init__(self, embed_dim):
        # 思路：由于不方便构建固定长度的多边形feature embedding，因此
        # 考虑在MergeNodeEncoder中加入一个图结构，将点上（或者三角形上）的特征聚合到
        # 多边形上。
        # 需求：
        # （1）三角形上的特征聚合到多边形上：提供的merge数据输入中，
        # 加入merge中最大面积的triangle和其它所有triangle的连接关系(感觉没有太大必要做这一点)
        # （2）多边形上的点特征聚合到多边形的内点上：向oriGraph的顶点中添加多边形的内点
        # 并建立这些内点和对应的多边形顶点的关系（先实现这个）
        # 方法：在venoGraph中提供merge2points的映射，
        super(MergeNodeEncoder, self).__init__(embed_dim)
        aux_dim_node_in = cfg.share.aux_dim_node * cfg.share.dim_pos
        self.febd_GCN = GCNConv(
            in_channels=aux_dim_node_in, out_channels=embed_dim, improved=True,
            add_self_loops=True
        )
        self.edgeExtra = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.oriNodeEmbed = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, batch):
        oriGraph, venoGraph, febdGraph = batch
        oriGraph = self.adjustOriGraphDualIdx(oriGraph)

        """计算特征, oriGraph.x通过GCN计算，包含了一层linear，所以删掉self.oriNode"""
        edge_extra = torch.cat([oriGraph.edge_angle, oriGraph.edge_type], dim=-1)
        edge_extra = self.edgeExtra(edge_extra)
        # edge_attr = self.oriEdgeBN(self.oriEdge(oriGraph.edge_attr))
        oriGraph.edge_attr = self.oriEdgeEmbed(oriGraph.edge_attr) + edge_extra
        oriGraph.edge_attr = self.oriEdge(oriGraph.edge_attr)

        x = self.febd_GCN(oriGraph.x, febdGraph.edge_index, None)
        x = self.oriNodeEmbed(x)

        """计算venoGraph.x"""
        veno_xs = []
        for low_ori, low, high in zip(
            febdGraph.meta_data['index_self_low'],
            febdGraph.meta_data['index_low'],
            febdGraph.meta_data['index_high']
        ):

            veno_xs.append(x[low: high])
        oriGraph.x = self.oriNode(x)
        veno_xs = torch.cat(veno_xs, dim=0)
        # mask = torch.logical_not(x == 0)
        # veno_xs = x[mask]
        assert veno_xs.shape[0] == venoGraph.x.shape[0], "wrong matching in venoGraph.x in Merge_Node"
        venoGraph.x = self.venoNode(veno_xs)
        venoGraph.edge_attr = self.venoEdge(self.venoEdgeEmbed(venoGraph.edge_attr))

        """替换venoGraph边特征"""
        vemap_idx = venoGraph.edge_dual_idx.squeeze()
        venoGraph.__setitem__('dual_attr', oriGraph.edge_attr[vemap_idx])

        """替换oriGraph的边特征"""
        edge_attr = venoGraph.edge_attr
        ones_padding = torch.ones(
            (1, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, ones_padding], dim=0)
        oemap_idx = oriGraph.edge_dual_idx.squeeze()
        oriGraph.__setitem__('dual_attr', edge_attr[oemap_idx])

        return oriGraph, venoGraph



@register_node_encoder('merge_node_resnet')
class MergeNodeResNetEncoder(MergeNodeEncoder):
    def __init__(self, embed_dim):
        super(MergeNodeResNetEncoder, self).__init__(embed_dim)
        self.imgEmbedding = ImgEncoderDict[cfg.gnn.imgEncoder]
        self.imgNorm = None
        self.drop_after_pos = torch.nn.Dropout(p=0.1)
        self.febd_GCN_img = MyGCNConv(in_channels=embed_dim, out_channels=embed_dim, improved=True)
        if cfg.share.imgEncoder_dim_out == embed_dim or cfg.share.imgEncoder_dim_out == -1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                in_channels=cfg.share.imgEncoder_dim_out, out_channels=embed_dim,
                kernel_size=(1, 1)
                ), nn.BatchNorm2d(embed_dim)
            )

    def forward(self, batch):
        oriGraph, venoGraph, febdGraph, imgT = batch
        oriGraph = self.adjustOriGraphDualIdx(oriGraph)
        oriVF = self.encodeImgFeat(batch)

        """计算特征"""
        edge_extra = torch.cat([oriGraph.edge_angle, oriGraph.edge_type], dim=-1)
        edge_extra = self.edgeExtra(edge_extra)
        # edge_attr = self.oriEdgeBN(self.oriEdge(oriGraph.edge_attr))
        oriGraph.edge_attr = self.oriEdgeEmbed(oriGraph.edge_attr) + edge_extra
        oriGraph.edge_attr = self.oriEdge(oriGraph.edge_attr)

        x = self.febd_GCN(oriGraph.x, febdGraph.edge_index)
        x = self.oriNodeEmbed(x)
        x_img = self.febd_GCN_img(oriVF, febdGraph.edge_index)
        x_fuse = self.drop_after_pos(x + x_img)

        """计算venoGraph.x, oriGraph.x"""
        veno_xs = []
        for low_ori, low, high in zip(
            febdGraph.meta_data['index_self_low'],
            febdGraph.meta_data['index_low'],
            febdGraph.meta_data['index_high']
        ):

            veno_xs.append(x_fuse[low: high])
        oriGraph.x = self.oriNode(x) # 只保留几何特征,使用self.oriNode()映射
        veno_xs = torch.cat(veno_xs, dim=0) # 同时包含几何特征和图像特征
        assert veno_xs.shape[0] == venoGraph.x.shape[0], "wrong matching in venoGraph.x in Merge_Node"
        venoGraph.x = self.venoNode(veno_xs)
        venoGraph.edge_attr = self.venoEdge(self.venoEdgeEmbed(venoGraph.edge_attr))

        """替换venoGraph边特征"""
        vemap_idx = venoGraph.edge_dual_idx.squeeze()
        venoGraph.__setitem__('dual_attr', oriGraph.edge_attr[vemap_idx])

        """替换oriGraph的边特征"""
        edge_attr = venoGraph.edge_attr
        ones_padding = torch.ones(
            (1, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, ones_padding], dim=0)
        oemap_idx = oriGraph.edge_dual_idx.squeeze()
        oriGraph.__setitem__('dual_attr', edge_attr[oemap_idx])

        return oriGraph, venoGraph


    def encodeImgFeat(self, batch):
        """计算图像特征"""
        oriGraph, venoGraph, febdGraph, imgT = batch
        imgF = self.imgEmbedding(imgT)
        if self.imgNorm is not None:
            imgF = self.imgNorm(imgF)
        if self.downsample is not None:
            imgF = self.downsample(imgF)
        imgF = F.interpolate(imgF, imgT.shape[2:], mode='bilinear', align_corners=True)
        b, c, h, w = imgF.shape
        imgF = imgF.view(b, c, -1)

        # 根据oriGraph和venoGraph中的节点，提取坐标(XY)
        oriVCoord = oriGraph.pos.to(torch.long)  #
        oriVCoord = oriVCoord[:, 1] * w + oriVCoord[:, 0]

        oriVF = []
        for i in range(b):
            oriMask = oriGraph.batch == i

            tempOriV = oriVCoord[oriMask].to(torch.long)
            tempOriVF = torch.index_select(imgF[i], dim=-1, index=tempOriV)

            oriVF.append(tempOriVF)

        oriVF = torch.cat(oriVF, dim=1).transpose(1, 0)
        return oriVF



@register_node_encoder('merge_node_resnet_1')
class MergeNodeResNetEncoder1(MergeNodeResNetEncoder):
    # 相比于MergeNodeResNetEncoder，本类修改了两个细节
    # 1.将edgeType的Norm和ReLU去掉，简化edge type到edge partition的信息通路。
    # 2.将图像特征加入oriGraph的节点特征中，利用语义信息辅助edge partition的预测。
    def __init__(self, embed_dim):
        super(MergeNodeResNetEncoder1, self).__init__(embed_dim)
        aux_dim_edge_in = cfg.share.aux_dim_edge * cfg.share.dim_pos
        # 第一步修改
        self.oriEdgeEmbed = nn.Linear(aux_dim_edge_in, embed_dim)
        self.edgeExtra = nn.Linear(4, embed_dim)
        del self.oriEdge

    def forward(self, batch):
        oriGraph, venoGraph, febdGraph, imgT = batch
        oriGraph = self.adjustOriGraphDualIdx(oriGraph)
        oriVF = self.encodeImgFeat(batch)

        """计算primal Graph的边特征"""
        edge_extra = torch.cat([oriGraph.edge_angle, oriGraph.edge_type], dim=-1)
        edge_extra = self.edgeExtra(edge_extra)
        # edge_attr = self.oriEdgeBN(self.oriEdge(oriGraph.edge_attr))
        oriGraph.edge_attr = self.oriEdgeEmbed(oriGraph.edge_attr) + edge_extra

        """计算venoGraph.x, oriGraph.x"""
        x = self.febd_GCN(oriGraph.x, febdGraph.edge_index)
        x = self.oriNodeEmbed(x)
        x_img = self.febd_GCN_img(oriVF, febdGraph.edge_index)
        x_fuse = self.drop_after_pos(x + x_img)

        veno_xs = []
        for low_ori, low, high in zip(
            febdGraph.meta_data['index_self_low'],
            febdGraph.meta_data['index_low'],
            febdGraph.meta_data['index_high']
        ):
            veno_xs.append(x_fuse[low: high])

        venoGraph.__setitem__('x_old', venoGraph.x.clone())

        # 第二步修改，将图像特征加到oriGraph的节点上
        oriGraph.x = self.oriNode(x + oriVF)
        veno_xs = torch.cat(veno_xs, dim=0) # 同时包含几何特征和图像特征
        assert veno_xs.shape[0] == venoGraph.x.shape[0], "wrong matching in venoGraph.x in Merge_Node"
        venoGraph.x = self.venoNode(veno_xs)

        """计算dual Graph的边特征"""
        venoGraph.edge_attr = self.venoEdge(self.venoEdgeEmbed(venoGraph.edge_attr))

        """替换venoGraph边特征"""
        vemap_idx = venoGraph.edge_dual_idx.squeeze()
        venoGraph.__setitem__('dual_attr', oriGraph.edge_attr[vemap_idx])
        """替换oriGraph的边特征"""
        edge_attr = venoGraph.edge_attr
        ones_padding = torch.ones(
            (1, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, ones_padding], dim=0)
        oemap_idx = oriGraph.edge_dual_idx.squeeze()
        oriGraph.__setitem__('dual_attr', edge_attr[oemap_idx])

        venoGraph.edge_attr = venoGraph.dual_attr.clone()

        return oriGraph, venoGraph
