import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet, VGG
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vgg import make_layers
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.graphgym.config import cfg


class VGGEncoder(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super().__init__(features, num_classes, init_weights)
        state_dict = torch.load('./models/vgg16_bn-torch.pth')
        self.load_state_dict(state_dict)
        del self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ResNetEncoder(ResNet):
    def __init__(
        self, block, layers,
            **kwargs
    ):
        super().__init__(block, layers, **kwargs)
        if issubclass(block, BasicBlock):
            state_dict = torch.load('./models/resnet34-torch.pth')
            # pass
        elif issubclass(block, Bottleneck):
            # state_dict = torch.load('./models/resnet50_v1c-2cccc1ad.pth')
            if layers[2] == 6:
                state_dict = torch.load('./models/resnet50-torch.pth')
            else:
                state_dict = torch.load('./models/resnet101-torch.pth')
            # pass
        else:
            raise NotImplementedError('wrong reset block and layer settings, check!')
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 输入图像是三通道
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 从layer1到layer4没有经过特征下采样
        output = x
        return output


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.imgEmbedding = torch.nn.Conv2d(
            in_channels=3, out_channels=256,
            kernel_size=(16, 16), stride=(8, 8)
        )
        self.norm = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.imgEmbedding(x)
        x = self.norm(x)
        x = self.ReLU(x)
        return x


ImgEncoderDict = {
    'resnet34': ResNetEncoder(BasicBlock, [3, 4, 6, 3]),
    'resnet50': ResNetEncoder(Bottleneck, [3, 4, 6, 3],
                    replace_stride_with_dilation=[False, True, True]),
    'resnet101': ResNetEncoder(Bottleneck, [3, 4, 23, 3],
                               replace_stride_with_dilation=[False, True, True]),
    'vgg16': VGGEncoder(
        make_layers(
            [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            batch_norm=True
        )
    ),
    'convblock': ConvBlock()
}


"""尝试使用位置向量作为图节点和边的特征"""
@register_node_encoder('dual_node')
class DualNodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        super(DualNodeEncoder, self).__init__()

        aux_dim_node_in = cfg.share.aux_dim_node * cfg.share.dim_pos
        aux_dim_edge_in = cfg.share.aux_dim_edge * cfg.share.dim_pos

        self.oriNode = torch.nn.Linear(
            embed_dim, embed_dim
        )
        self.oriEdge = torch.nn.Linear(
            embed_dim, embed_dim
        )
        # self.oriNodeBN = torch.nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.1)
        # self.oriEdgeBN = torch.nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.1)

        self.venoNode = torch.nn.Linear(embed_dim, embed_dim)
        self.venoEdge = torch.nn.Linear(embed_dim, embed_dim)
        # self.venoNodeBN = torch.nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.1)
        # self.venoEdgeBN = torch.nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.1)

        # 标准化重新加feature embedding；
        # 原则：如果只使用位置向量，则后面跟一层linear norm act:
        # 如果和图像特征配合，不需要feature embedding
        self.oriNodeEmbed = nn.Sequential(
            nn.Linear(aux_dim_node_in, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=True),
            nn.ReLU(inplace=True)
        )
        self.oriEdgeEmbed = nn.Sequential(
            nn.Linear(aux_dim_edge_in, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=True),
            nn.ReLU(inplace=True)
        )
        self.venoNodeEmbed = nn.Sequential(
            nn.Linear(cfg.share.dim_node_in, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=True),
            nn.ReLU(inplace=True)
        )
        self.venoEdgeEmbed = nn.Sequential(
            nn.Linear(cfg.share.dim_edge_in, embed_dim),
            nn.LayerNorm(embed_dim, elementwise_affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, batch):
        oriGraph, venoGraph = batch
        oriGraph = self.adjustOriGraphDualIdx(oriGraph)
        """计算特征"""
        oriGraph.x = self.oriNode(self.oriNodeEmbed(oriGraph.x))
        oriGraph.edge_attr = self.oriEdge(self.oriEdgeEmbed(oriGraph.edge_attr))

        venoGraph.x = self.venoNode(self.venoNodeEmbed(venoGraph.x))
        venoGraph.edge_attr = self.venoEdge(self.venoEdgeEmbed(venoGraph.edge_attr))

        """替换venoGraph边特征"""
        vemap_idx = venoGraph.edge_dual_idx.squeeze()
        venoGraph.__setitem__('dual_attr', oriGraph.edge_attr[vemap_idx])

        """
        替换oriGraph的边特征，存在某些oriGraph的边，在venoGraph中找不到对偶边。
        对于这种情况，给venoGraph的edge attr后面concat一个padding，所有找不到
        对偶边的都使用padding的特征。
        """
        edge_attr = venoGraph.edge_attr
        ones_padding = torch.ones(
            (1, edge_attr.shape[1]), dtype=torch.float, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, ones_padding], dim=0)
        oemap_idx = oriGraph.edge_dual_idx.squeeze()
        oriGraph.__setitem__('dual_attr', edge_attr[oemap_idx])

        return oriGraph, venoGraph

    def adjustOriGraphDualIdx(self, oriGraph):
        oemap_idx = oriGraph.edge_dual_idx
        oemap_idx[oemap_idx < 0] = -1
        oriGraph.edge_dual_idx = oemap_idx
        return oriGraph


@register_node_encoder('edge_extra')
class EdgeExtraEncoder(DualNodeEncoder):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)
        self.edgeExtra = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch):
        oriGraph, venoGraph = batch
        oriGraph = self.adjustOriGraphDualIdx(oriGraph)
        """计算特征，将edge的额外属性(角度、是否为墙体)加入到输入中"""
        oriGraph.x = self.oriNode(self.oriNodeEmbed(oriGraph.x))
        edge_extra = torch.cat([oriGraph.edge_angle, oriGraph.edge_type], dim=-1)
        edge_extra = self.edgeExtra(edge_extra)
        # edge_attr = self.oriEdgeBN(self.oriEdge(oriGraph.edge_attr))
        oriGraph.edge_attr = self.oriEdgeEmbed(oriGraph.edge_attr) + edge_extra
        oriGraph.edge_attr = self.oriEdge(oriGraph.edge_attr)

        venoGraph.x = self.venoNode(self.venoNodeEmbed(venoGraph.x))
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

@register_node_encoder('dual_node_img')
class DualNodeIMGEncoder(EdgeExtraEncoder):
    def __init__(self, embed_dim):
        super(DualNodeIMGEncoder, self).__init__(embed_dim)
        self.imgEmbedding = torch.nn.Conv2d(
            in_channels=3, out_channels=embed_dim,
            kernel_size=(16, 16), stride=(8, 8)
        )
        self.imgNorm = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.drop_after_pos = torch.nn.Dropout(p=0.1)

        if cfg.share.imgEncoder_dim_out == embed_dim or cfg.share.imgEncoder_dim_out == -1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                in_channels=cfg.share.imgEncoder_dim_out, out_channels=embed_dim,
                kernel_size=(1, 1)
                ), nn.BatchNorm2d(embed_dim)
            )

    def encodeImgFeat(self, batch):
        """计算图像特征"""
        oriGraph, venoGraph, imgT = batch
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

        numN, dimN = venoGraph.pos.shape
        numPts_perN = dimN // 2
        venoVCoord = venoGraph.pos.to(torch.long)  # num_node * n, 2
        # venoVCoord = venoVCoord[:, 1] * w + venoVCoord[:, 0]

        oriVF, venoVF = [], []
        for i in range(b):
            oriMask = oriGraph.batch == i
            venoMask = venoGraph.batch == i

            tempOriV = oriVCoord[oriMask].to(torch.long)
            tempOriVF = torch.index_select(imgF[i], dim=-1, index=tempOriV)

            tempVenoV = venoVCoord[venoMask].to(torch.long).view(-1, 2)
            tempVenoV = tempVenoV[:, 1] * w + tempVenoV[:, 0]
            tempVenoVF = torch.index_select(imgF[i], dim=-1, index=tempVenoV)
            assert tempVenoVF.shape[1] % numPts_perN == 0, \
                'wrong points group partitioned for node, check the node.'
            if numPts_perN > 1:
                tempVenoVF = F.avg_pool1d(
                    tempVenoVF.unsqueeze(0), kernel_size=numPts_perN, stride=numPts_perN
                )

            oriVF.append(tempOriVF)
            venoVF.append(tempVenoVF.squeeze(0))

        oriVF, venoVF = torch.cat(oriVF, dim=1).transpose(1, 0), \
                        torch.cat(venoVF, dim=1).transpose(1, 0)
        eIdx1, eIdx2 = oriGraph.edge_index[0], oriGraph.edge_index[1]
        eF1, eF2 = oriVF[eIdx1], oriVF[eIdx2]
        eF = (eF1 + eF2) / 2.

        return venoVF, eF

    def forward(self, batch):
        oriGraph, venoGraph, imgT = batch
        oriGraph = self.adjustOriGraphDualIdx(oriGraph)
        venoVF, eF = self.encodeImgFeat(batch)

        """计算特征,oriGraph和EdgeExtra中的计算方式一样；
        venoGraph的节点中加入图像特征
        """
        oriGraph.x = self.oriNode(self.oriNodeEmbed(oriGraph.x))
        edge_extra = torch.cat([
            oriGraph.edge_angle, oriGraph.edge_type
        ], dim=-1)
        edge_extra = self.edgeExtra(edge_extra)
        # edge_attr = self.oriEdgeBN(self.oriEdge(oriGraph.edge_attr))
        oriGraph.edge_attr = self.oriEdgeEmbed(oriGraph.edge_attr) + edge_extra
        oriGraph.edge_attr = self.oriEdge(oriGraph.edge_attr)

        venoGraph.x = self.venoNodeEmbed(venoGraph.x)
        venoGraph.x = self.venoNode(self.drop_after_pos(venoGraph.x + venoVF))

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


@register_node_encoder('dual_node_resnet')
class DualNodeResNetEncoder(DualNodeIMGEncoder):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)

        self.imgEmbedding = ImgEncoderDict[cfg.gnn.imgEncoder]
        self.imgNorm = None


if __name__ == "__main__":
    from torchstat import stat
    models = ImgEncoderDict['resnet101']
    # input = torch.randn(1, 3, 512, 512)
    stat(models, (3, 512, 512))