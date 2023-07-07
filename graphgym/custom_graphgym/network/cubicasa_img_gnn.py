import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.models.head  # noqa, register module

from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg


class ResNetEncoder(ResNet):
    def __init__(
        self, block, layers,
            **kwargs
    ):
        super().__init__(block, layers, **kwargs)

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
        # x = self.layer4(x)
        # 从layer1到layer4没有经过特征下采样
        output = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

        return output


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=5),  # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2, padding=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, padding=1),  # conv3
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2, padding=1),  # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2, padding=1)  # conv5
        )

        self._initialize_weight()

    def _initialize_weight(self):
        """
        initialize network parameters
        """
        tmp_layer_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tmp_layer_idx = tmp_layer_idx + 1
                if tmp_layer_idx < 6:
                    # kaiming initialization
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                else:
                    # initialization for adjust layer as in the original paper
                    m.weight.data.fill_(1e-3)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.feat_extraction(x)
        output = F.interpolate(x, (h, w), mode='bilinear')
        return output


class AlexNetXPadding(AlexNet):
    def __init__(self):
        super(AlexNetXPadding, self).__init__()
        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),  # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),  # conv3
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, groups=2),  # conv4
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, groups=2)  # conv5
        )

        self._initialize_weight()


class AlexNetCut(AlexNet):
    def __init__(self):
        super(AlexNetCut, self).__init__()
        # architecture (AlexNet like)
        self.feat_extraction = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),  # conv1
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, groups=2),  # conv2, group convolution
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self._initialize_weight()


ImgEncoderDict = {
    'resnet': ResNetEncoder(BasicBlock, [3, 4, 6, 3]),
    'AlexNet_xPadding': AlexNetXPadding(),
    'AlexNet': AlexNet(),
    'AlexNet_cut': AlexNetCut()
}


@register_network('CUBICASA_IMG')
class CubicasaImgGNN(GNN):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.imgEncoder = ImgEncoderDict[cfg.gnn.imgEncoder]
        if isinstance(self.imgEncoder, ResNetEncoder):
            state_dict = torch.load('./models/resnet34-torch.pth')
            self.imgEncoder.load_state_dict(state_dict)

        # self.encoderAlign = nn.Linear(256 * BasicBlock.expansion, cfg.gnn.dim_inner)

    def getNodeFeat(self, batch, imgFeat):
        node = batch.pos
        nodeBIdx = batch.batch.squeeze(-1)

        numNode, nodeDim = node.shape
        nodePointsNum = nodeDim // 2 # avgPooling的stride和kernel
        bSz, c, h, w = imgFeat.shape

        # 为了实现方便，这里先不要应用feature align(RoI Align)的写法
        nodePos = node.to(torch.int)
        imgFeat = imgFeat.view(bSz, c, -1) # hw coord

        nodeFeats = []
        for i in range(bSz):
            tempNode = nodePos[nodeBIdx == i] # XY coord,
            tempNode = tempNode.view(-1, 2)

            indices = tempNode[:, 1] * w + tempNode [:, 0] # XY -> HW
            nodeFeat = torch.index_select(imgFeat[i], dim=1, index=indices.to(torch.long)) # featdim, len(indices)
            # 如果index_select()中的indices有越界的情况，则可能会报错 Unable to get repr for＜class‘torch.Tensor‘＞
            # 之后对tensor进行运算时，会报错CUDA error: device-side assert triggered
            assert nodeFeat.shape[1] % nodePointsNum == 0, \
                'wrong points group partitioned for node, check the node.'
            nodeFeat = F.avg_pool1d(
                nodeFeat.unsqueeze(0), kernel_size=nodePointsNum, stride=nodePointsNum
            )
            nodeFeats.append(nodeFeat)

        nodeFeats = torch.cat(nodeFeats, dim=-1).permute(2, 1, 0).squeeze(-1) # len(indices), featdim
        # nodeFeats = self.encoderAlign(nodeFeats)
        batch.x = nodeFeats
        return batch

    def forward(self, batch):
        fplanVertexGraph, fplanVenoGraph, fplanImg = \
            batch[0], batch[1], batch[2]

        fplanFeat = self.imgEncoder(fplanImg)
        batch = self.getNodeFeat(fplanVenoGraph, fplanFeat)

        for name, module in self.named_children():
            if name not in ['imgEncoder', 'encoderAlign']:
                batch = module(batch)
            else:
                pass

        return batch

