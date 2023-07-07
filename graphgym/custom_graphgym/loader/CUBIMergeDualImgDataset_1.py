import os
from typing import Any

import torchvision
import numpy as np
import torch
import pickle

from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import *
from torch_geometric.graphgym.register import register_loader, register_dataset
from torch_geometric.graphgym.contrib.transform.verification_transforms import GraphRandomFlip, \
    RandomHorizontalFlip, RandomVerticalFlip
from torch_geometric.graphgym.contrib.transform.graph_transform import \
    AttrbuteReplace, SinePosEncoding, AdaptiveTranslation, EdgeTypeOneHot, \
    MyToUndirected


class DualGraphData(Data):

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """处理对偶图的标号时，返回的应该是对偶图的点和边的数量"""
        assert isinstance(self.meta_data, dict), "there must be an attribute meta_data."
        assert self.meta_data.__contains__('dual_num_edges'), \
            'there must be num dual edges in the meta_data.'
        assert self.meta_data.__contains__('dual_num_nodes'), \
            'there must be num dual nodes in the meta_data.'

        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        elif key == 'edge_dual':
            return self.meta_data['dual_num_nodes']
        elif key == 'edge_dual_idx':
            return self.meta_data['dual_num_edges']
        else:
            return 0

@register_loader('CUBICASA_Merge_DUALIMG_1')
def load_dataset_CUBICASA(format, name, dataset_dir):
    if format == 'PyG':
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        imgTransform = torchvision.transforms.Compose([
            ToTensor(),
            Resize(size=256),
            Normalize(norm_mean, norm_std)
        ])
        if cfg.dataset.augmentation:
            aug = [
                GraphRandomFlip(attr=['x', 'edge_attr', 'pos']),
                GraphRandomFlip(attr=['x', 'edge_attr', 'pos'], flip_mode='vertical')
            ]
            img_aug = [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
            ]
        else:
            aug = None
            img_aug = None
        dataset_train = CUBIMergeDualIMGFloorplan_1(
            rootDir=dataset_dir, split='train', augmentation=aug,
            img_transform=imgTransform, img_augmentation=img_aug
        )
        dataset_val = CUBIMergeDualIMGFloorplan_1(rootDir=dataset_dir, split='val', img_transform=imgTransform)
        dataset_test = CUBIMergeDualIMGFloorplan_1(rootDir=dataset_dir, split='test', img_transform=imgTransform)
        return [dataset_train, dataset_val, dataset_test]
        # return [dataset_val, dataset_test]
    else:
        raise NotImplementedError


@register_dataset('CUBICASA_Merge_DUALIMG_1')
class CUBIMergeDualIMGFloorplan_1(InMemoryDataset):
    """
    加入了febdGraph，目的是丰富每一个多边形的特征表达。
    """
    CLASSES = (
        'background', 'Outdoor', 'Wall', 'Kitchen', 'Dining', 'Bedroom',
        'Bath', 'Entrance', 'Railing', 'Closet', 'Garage', 'Corridor',
    )

    def __init__(
        self, rootDir,
        split='train',
        transform=None,
        augmentation=None,
        img_transform=None,
        img_augmentation=None,
        img_to_float32=True,
        dropEdge=0
    ):
        super(CUBIMergeDualIMGFloorplan_1, self).__init__()

        MAPPING_SHAPE = 256
        with open(os.path.join(rootDir, split + '.txt'), 'r') as f:
            self.imageFolders = f.readlines()
        excludeList = [
            '/high_quality_architectural/2003/\n',
            '/high_quality_architectural/2565/\n',
            '/high_quality_architectural/6143/\n',
            '/high_quality_architectural/10074/\n',
            '/high_quality_architectural/10754/\n',
            '/high_quality_architectural/10769/\n',
            '/high_quality_architectural/14611/\n',
            '/high_quality/7092/\n',
            '/high_quality/1692/\n',

            '/high_quality_architectural/10/\n',  # img does not match label
        ]
        self.imageFolders = list(set(self.imageFolders).difference(set(excludeList)))

        self.rootDir, self.partition = rootDir, split
        self.triFplans = self._loadProcessedData()
        if transform is not None:
            self.transform = transform
        else:
            self.transform = [
                Compose([
                    # fplanVertexGraph
                    ToUndirected(),
                    AdaptiveTranslation(img_shape=MAPPING_SHAPE, attr=['x', 'edge_attr', 'pos']),
                    SinePosEncoding(
                        embed_dim=cfg.share.dim_pos // 2,
                        attr=['x', 'edge_attr'],
                        img_shape=MAPPING_SHAPE
                    ),
                    EdgeTypeOneHot()
                ]),
                Compose([
                    # ToUndirected()会把所有第一维和num_edges一致的
                    # 属性都当成是边的属性,并undirected化
                    # fplanVenoGraph
                    AttrbuteReplace(src='x', tgt='pos'),
                    ToUndirected(),
                    AdaptiveTranslation(img_shape=MAPPING_SHAPE, attr=['x', 'edge_attr', 'pos']),
                    SinePosEncoding(
                        embed_dim=cfg.share.dim_pos // 2,
                        attr=['x', 'edge_attr'],
                        img_shape=MAPPING_SHAPE
                    ),
                    EdgeTypeOneHot(),
                    MyToUndirected()
                ]),
                Compose([
                    AdaptiveTranslation(img_shape=MAPPING_SHAPE),
                    SinePosEncoding(
                        attr=['x', 'edge_attr'],
                        img_shape=MAPPING_SHAPE
                    )
                ])
            ]
        self.augmentation = augmentation
        self.img_transform = img_transform
        self.img_augmentation = img_augmentation
        self.img_to_float32 = img_to_float32

    def __len__(self):
        return len(self.imageFolders)

    def __getitem__(self, item):
        # 在父类中，__getitem__() 实现了 get() 方法，感觉实际上并不需要
        imgName = self.imageFolders[item].split('/')[-2]
        if self.pre_transform is not None:
            # 自行实现的，对原始图数据的修正
            res = self.pre_transform(self.triFplans[imgName])
        else:
            res = self.triFplans[imgName]

        img = Image.open(
            os.path.join(
                self.rootDir, 'img_dir', self.partition, '{}.png'.format(imgName)
            )
        )
        imgArray = np.array(img)
        if self.img_to_float32:
            imgArray = imgArray.astype(np.float32)
            imgArray = imgArray[:, :, :3] / 255.

        # 要根据原始图结构构建图模型，直觉上讲，可以构建两种图模型，
        # 第一种是以户型图的墙体节点为node，边为edge;
        fplanVertexGraph, fplanVenoGraph, febdGraph = self.genGraph(res, imgName)

        if self.img_transform is not None:
            imgT = self.img_transform(imgArray)
        else:
            raise RuntimeError('Please checkout the image transforms.')

        if self.augmentation is not None:
            assert self.img_augmentation is not None, \
                'self.img_augmentation must not be None when self.augmentation' \
                'is not None.'
            for aug, img_aug in zip(self.augmentation, self.img_augmentation):
                p = np.random.rand(1)
                fplanVertexGraph = aug(fplanVertexGraph, p)
                fplanVenoGraph = aug(fplanVenoGraph, p)
                febdGraph = aug(febdGraph, p)
                imgT = img_aug(imgT, p)

        if self.transform is not None:
            assert isinstance(self.transform, list), \
                'self.transform must contain 3 transforms, for ' \
                'fplanVertexGraph, fplanVenoGraph, fplanDualGraph, ' \
                'respectively.'
            # 对两个图做不同的变换
            fplanVertexGraph = self.transform[0](fplanVertexGraph)
            fplanVenoGraph = self.transform[1](fplanVenoGraph)
            febdGraph = self.transform[2](febdGraph)

        # fplanDualGraph = self.genFplanAuxGraph(fplanVertexGraph)
        # if self.transform is not None:
        #     fplanDualGraph = self.transform[2](fplanDualGraph)
        O2VT, V2OT = self.dualEdgeMapping(fplanVertexGraph, fplanVenoGraph)
        fplanVertexGraph['edge_dual_idx'], fplanVenoGraph['edge_dual_idx'] = O2VT, V2OT

        fplanVertexGraph['meta_data']['dual_num_edges'] = fplanVenoGraph.num_edges
        fplanVertexGraph['meta_data']['dual_num_nodes'] = fplanVenoGraph.num_nodes
        fplanVenoGraph['meta_data']['dual_num_edges'] = fplanVertexGraph.num_edges
        fplanVenoGraph['meta_data']['dual_num_nodes'] = fplanVertexGraph.num_nodes

        return fplanVertexGraph, fplanVenoGraph, febdGraph, imgT

    def _loadProcessedData(self):
        # 若包含多边形内部点，则使用V10；若只包含边界点，则使用V9
        with open(os.path.join(
                self.rootDir, 'merge_' + self.partition + '_phase_V10.pkl'
        ), 'rb') as f:
            oriFplans = pickle.load(f)
        return oriFplans

    def dualEdgeMapping(self, fpOGraph, fpVGraph):
        Oe, Oem = fpOGraph['edge_index'], fpOGraph['edge_dual'] # original edge mapping
        Ve, Vem = fpVGraph['edge_index'], fpVGraph['edge_dual'] # veno edge mapping
        listOe, listOem = Oe.numpy().transpose(1, 0).tolist(), Oem.numpy().tolist()
        listVe, listVem = Ve.numpy().transpose(1, 0).tolist(), Vem.numpy().tolist()

        # 将一对无向边的对偶边也调成无向边的形式
        for idx, e in enumerate(listOe):
            if e[0] > e[1]:
                temp = listOem[idx]
                listOem[idx] = [temp[1], temp[0]]
        for idx, e in enumerate(listVe):
            if e[0] > e[1]:
                temp = listVem[idx]
                listVem[idx] = [temp[1], temp[0]]

        # 第0维 veno idx; 第1维 ori idx
        V2O, O2V = [], [] # veno idx -> original idx, veno: 0,1,2,3,4,...; original: o0,o1,o2,o3,o4,...
        # 用于将oriGraph中的边的属性赋给venoGraph
        for idx, em in enumerate(listVem):
            emIdx = listOe.index(em)
            V2O.append(emIdx)
        tV2O = torch.tensor(V2O, dtype=torch.long).unsqueeze(1)

        for idx, em in enumerate(listOem):
            if -1 not in em:
                emIdx = listVe.index(em)
            else:
                emIdx = -1 * 10000
            O2V.append(emIdx)
        tO2V = torch.tensor(O2V, dtype=torch.long).unsqueeze(1)

        return tO2V, tV2O

    def genGraph(self, dataElem, elemName):
        """
        whole edge sets <=> {有对偶边的边集|A1} + {无对偶边的边集(边界)|A2}
                        <=> {墙体|B1} + {非墙体|B2}
                        <=> {分隔区域的边集|C1} + {不分隔区域的边集|C2}
        :param dataElem:
        :param elemName:
        :return:
        """
        """第一种是以户型图为单位，构建原始图，提取原始数据中的各个元素"""
        fplanGraphDict = dataElem[0]
        nodePos = fplanGraphDict['vertices']
        edgeIndices = np.array(fplanGraphDict['edge'])
        edgeAttr1, edgeAttr2 = nodePos[edgeIndices[:, 0]], nodePos[edgeIndices[:, 1]]  # 边特征是节点的坐标位置向量
        edgeAttr = np.concatenate([edgeAttr1, edgeAttr2], axis=-1)  # edgenum, 4
        edgelen = np.linalg.norm(edgeAttr1 - edgeAttr2, axis=-1) + 1e-7
        edgeSub = edgeAttr1 - edgeAttr2
        edgeSine, edgeCos = edgeSub[:, 1] / edgelen, edgeSub[:, 0] / edgelen
        scale = np.ones_like(edgeSine)
        scale[edgeCos < 0] = -1

        edgeSine, edgeCos = edgeSine * scale, edgeCos * scale
        edgeAngle = np.concatenate([edgeSine[:, np.newaxis], edgeCos[:, np.newaxis]], axis=-1)

        # 生成tensor
        nodePosT = torch.tensor(nodePos, dtype=torch.float)
        edgeT = torch.transpose(
            torch.tensor(edgeIndices, dtype=torch.long), 0, 1
        )  # 2,edgenum
        edgeAttrT = torch.tensor(edgeAttr, dtype=torch.float)
        temp = torch.tensor(fplanGraphDict['edge_attr'])
        edgeTypeT, edgeLabelT = temp[:, 0].unsqueeze(-1), temp[:, 1].unsqueeze(-1)  # edgenum,1
        edgeIDT = torch.arange(0, edgeTypeT.shape[0], dtype=torch.int).unsqueeze(-1)
        edgeDualT = torch.tensor(fplanGraphDict['edge_dual'], dtype=torch.long)
        edgeAngleT = torch.tensor(edgeAngle, dtype=torch.float)

        febd_indices = np.array(fplanGraphDict['inner2bd_index'])
        febd_attr1, febd_attr2 = nodePos[febd_indices[:, 0]], nodePos[febd_indices[:, 1]]
        febd_attr = np.concatenate([febd_attr1, febd_attr2], axis=-1)
        febd_attrT = torch.tensor(febd_attr, dtype=torch.float)
        febdT = torch.transpose(
            torch.tensor(febd_indices, dtype=torch.long), 0, 1
        )

        del nodePos, edgeIndices, edgeAttr1, edgeAttr2, edgeAttr, edgeAngle
        del febd_indices, febd_attr1, febd_attr2
        # 边是户型图的结构
        fplanVertexGraph = DualGraphData(
            x=nodePosT.clone(), pos=nodePosT.clone(),
            edge_index=edgeT, edge_attr=edgeAttrT, edge_dual=edgeDualT,
            edge_angle=edgeAngleT,
            edge_type=edgeTypeT, edge_label=edgeLabelT,
            # edge_id=edgeIDT
            meta_data={
                'dual_num_edges': 0,
                'dual_num_nodes': 0
            },
        )

        # 为了建立多边形primitive的顶点，到它的中心的关系图
        febdGraph = Data(
            x=nodePosT.clone(), pos=nodePosT.clone(),
            edge_index=febdT, edge_attr=febd_attrT,
            meta_data={
                'index_self_low': torch.min(febdT[0]).item(),
                'index_low': torch.min(febdT[1]).item(),
                'index_high': torch.max(febdT[1]).item() + 1
            }
            # x表示节点的特征，是节点坐标的位置向量
            # pos表示节点的坐标
        )
        del nodePosT, edgeT, edgeAttrT, edgeDualT, edgeTypeT, edgeLabelT, edgeIDT

        """第二种是以三角剖分为图节点，A1集的对偶边作为图边，由于三角剖分的维诺图"""
        fplanVenoDict = dataElem[1]
        tNodePosT = torch.tensor(fplanVenoDict['vertices'], dtype=torch.float)
        tNodeAttrT = torch.tensor(fplanVenoDict['x'], dtype=torch.float)
        tEdgeT = torch.transpose(
            torch.tensor(fplanVenoDict['segments'], dtype=torch.long), 0, 1
        )
        tEdgeAttrT = torch.tensor(fplanVenoDict['segment_attr'], dtype=torch.float)
        tLabel = torch.tensor(fplanGraphDict['triangles_label'], dtype=torch.long)

        tEdgeTypeT = torch.tensor(
            fplanVenoDict['segments_type'], dtype=torch.float
        ).unsqueeze(-1)
        tEdgeDualT = torch.tensor(fplanVenoDict['segments_dual'], dtype=torch.long)

        # 计算额外的node 属性，包括，三角形面积
        # 以一个户型图为单位，将每一个节点的面积归一化（所有节点面积和为1）
        nodeArea = torch.tensor(fplanGraphDict['triangles_area'], dtype=torch.float)
        nodeAreaSum = torch.sum(nodeArea)
        nodeArea = nodeArea / nodeAreaSum

        # 给fplanVenoGraph增加dilation 为2、4edge index，其实就是邻接矩阵的2、4次相乘
        adjT = self.edgeIndex2Adj(tEdgeT)
        tEdgeT_judge = self.Adj2EdgeIndex(adjT)
        # 转换为Adj，再转换回edge_index，不能保持原来edge_index的顺序

        adjT_d2 = torch.mm(adjT, adjT)
        adjT_d4 = torch.mm(adjT_d2, adjT_d2)
        adjT_d8 = torch.mm(adjT_d4, adjT_d4)
        adjT_d2 = self.removeRepeated(adjT_d2, adjT)
        adjT_d4 = self.removeRepeated(adjT_d4, adjT_d2 + adjT)
        adjT_d8 = self.removeRepeated(adjT_d8, adjT + adjT_d2 + adjT_d4)

        # adjT_d2, adjT_d4, adjT_d8 = adjT_d2 - adjT, adjT_d4 - adjT, adjT_d8 - adjT
        tEdgeT_d2, tEdgeT_d4, tEdgeT_d8 = self.Adj2EdgeIndex(adjT_d2), \
                                          self.Adj2EdgeIndex(adjT_d4), \
                                          self.Adj2EdgeIndex(adjT_d8)

        # 边是三角剖分的相邻关系
        fplanVenoGraph = DualGraphData(
            x=tNodeAttrT, pos=tNodePosT, y=tLabel,
            x_extra=nodeArea,
            edge_index=tEdgeT, edge_attr=tEdgeAttrT,
            edge_type=tEdgeTypeT, edge_dual=tEdgeDualT,
            edge_index_d2=tEdgeT_d2, edge_index_d4=tEdgeT_d4, edge_index_d8=tEdgeT_d8,
            meta_data={
                'filename': elemName,
                'node_area': fplanGraphDict['triangles_area'],
                'dual_num_edges': 0,
                'dual_num_nodes': 0,
            },
            # pos表示了三角形内点的坐标值，需要转换后才能对应到图像中
            # edge_dual 保存了当前的边在fplanVertexGraph中的对应边
        )
        del tNodeAttrT, tNodePosT, tLabel, tEdgeT, tEdgeAttrT, tEdgeTypeT, tEdgeDualT
        return fplanVertexGraph, fplanVenoGraph, febdGraph

    def edgeIndex2Adj(self, edge_index):
        """
        规定adj的维度0表示edge_index的维度0，维度1表示edge_index的维度1
        :param edge_index: (tensor)
        :return:
        """
        minVIdx, maxVIdx = 0, torch.max(edge_index) - torch.min(edge_index)
        adj = torch.zeros([maxVIdx + 1, maxVIdx + 1], dtype=torch.int).view(-1)
        adj_coord = edge_index[0] * (maxVIdx + 1) + edge_index[1]

        adj[adj_coord] = 1  # 单向的，需要转换成无向图
        adj = adj.view(maxVIdx + 1, maxVIdx + 1)
        adj = adj + torch.transpose(adj, 0, 1)
        return adj # 对角线为0

    def Adj2EdgeIndex(self, adj):
        # adj是无向的，应该转为单向的，所以先把adj转换为上三角矩阵
        h, w = adj.shape
        adj = torch.triu(adj, diagonal=1)
        coord = torch.arange(0, h * w, step=1)
        mask = adj.view(-1) > 0
        index_flatten = coord[mask]
        h_coord = torch.div(index_flatten, w, rounding_mode='trunc')
        w_coord = index_flatten - h_coord * w
        edge_index = torch.cat([h_coord.unsqueeze(0), w_coord.unsqueeze(0)], dim=0)
        return edge_index

    def removeRepeated(self, adj_1, adj_2):
        adj_1, adj_2 = adj_1.to(torch.bool), adj_2.to(torch.bool)
        adj_1 = torch.logical_and(adj_1, torch.logical_not(adj_2))
        adj_1 = adj_1.to(torch.int)
        return adj_1

    def genFplanAuxGraph(self, vertexGraph):
        """
        abandoned
        放在vertexGraph经过self.transform处理之后再调用
        对vertexGraph的transform应该只包含ToUndirected()
        :param vertexGraph: 以原始户型图为节点的图数据 (V, E)
        :return: 以E作为节点V'，具有相同端点的边互相连通，作为边E'，fplanDualGraph
            x: fplanVertexGraph的边特征;
            x_type: fplanVertexGraph的边属性;
            pos:fplanVertexGraph的边中心位置;
            edge_index:给fplanVertexGraph中的每一条边一个id，若两个边共享同一个端点，
            则他们的id组合后，在fplanDualGraph中就是一条边

        所有folanVertexGraph中的连接，都以connected开头

        """
        connectedV = vertexGraph['edge_index'][0, :]
        edgeIndexes, edgeAttrs = [], []
        for i in range(vertexGraph['x'].shape[0]):
            edgeID_i = vertexGraph['edge_id'][connectedV == i].squeeze(-1)
            edgeIdx1, edgeIdx2 = torch.meshgrid(edgeID_i, edgeID_i)
            filterT = torch.ones_like(edgeIdx1) - torch.eye(edgeIdx1.shape[0])
            filterT = filterT.view(1, -1)

            edgeIdx1 = edgeIdx1.contiguous().view(1, -1)[filterT == 1].unsqueeze(0)
            edgeIdx2 = edgeIdx2.contiguous().view(1, -1)[filterT == 1].unsqueeze(0)
            edgeIndex = torch.cat([edgeIdx1, edgeIdx2], dim=0) # 2,numEdge
            edgeAttr = vertexGraph['pos'][i].unsqueeze(0) # 1,numDim
            edgeAttr = edgeAttr.repeat(edgeIndex.shape[1], 1) # numEdge,numDim

            edgeIndexes.append(edgeIndex)
            edgeAttrs.append(edgeAttr)

        edgeIndexes = torch.cat(edgeIndexes, dim=1)
        edgeAttrs = torch.cat(edgeAttrs, dim=0)

        connectedEdgeAttr = vertexGraph['edge_attr']
        nodePosT = (connectedEdgeAttr[:, :2] + connectedEdgeAttr[:, 2:]) / 2
        fplanDualGraph = Data(
            x=connectedEdgeAttr,
            x_type=vertexGraph['edge_type'],
            pos=nodePosT,
            edge_attr=edgeAttrs,
            edge_index=edgeIndexes,
            y=vertexGraph['edge_label']
        )
        return fplanDualGraph


if __name__ == "__main__":
    with open('/home/ybc2021/Datasets/CUBI_CAD/train_phase.pkl', 'rb') as f:
        data = pickle.load(f)
    print()
    # transform = Compose([
    #     ToUndirected(),
    #     NormalizeScale()
    # ])
    trainDataset = CUBIMergeDualIMGFloorplan_1(rootDir='/home/ybc2021/Datasets/CUBI_CAD',
                                 transform=None)
    # LightningDataset(
    #     train_dataset=trainDataset,
    #     batch_size=4
    # )
