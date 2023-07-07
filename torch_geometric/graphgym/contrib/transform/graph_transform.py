from typing import List, Optional, Tuple, Union, Any

import math
import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce


def to_undirected(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
    edge_attr_t = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will remove duplicates for all its entries.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    # Maintain backward compatibility to `to_undirected(edge_index, num_nodes)`
    if isinstance(edge_attr, int):
        edge_attr = None
        num_nodes = edge_attr

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    if edge_attr is not None and isinstance(edge_attr, Tensor):
        edge_attr = torch.cat([edge_attr, edge_attr_t(edge_attr)], dim=0)
    elif edge_attr is not None:
        edge_attr = [torch.cat([e, t(e)], dim=0) for t, e in zip(edge_attr_t, edge_attr)]

    return coalesce(edge_index, edge_attr, num_nodes, reduce)


@functional_transform('my_to_undirected')
class MToUndirected(BaseTransform):
    r"""
    same to the official implementation of ToUndirected, but different in
    deal with edge attr.
    """
    def __init__(self, reduce: str = "add", merge: bool = True, edge_attr_t = None):
        self.reduce = reduce
        self.merge = merge
        self.edge_attr_t = edge_attr_t

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key

                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = torch.stack([col, row], dim=0)

                inv_store = data[dst, f'rev_{rel}', src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values,
                    self.edge_attr_t, reduce=self.reduce
                )

                for key, value in zip(keys, values):
                    store[key] = value

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@functional_transform('attribute_replace')
class AttrbuteReplace(BaseTransform):
    def __init__(self, src=None, tgt=None):
        self.src = src
        self.tgt = tgt

    def __call__(self, data):
        for store in data.stores:
            store[self.tgt] = store[self.src]

        return data


@functional_transform('sinepos_encoding')
class SinePosEncoding(BaseTransform):
    """
    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """
    def __init__(
            self, embed_dim=64, img_shape=1000,
            temperature=10000,
            normalize=False, scale=None,
            attr=None
    ):
        if attr is None:
            attr = ['x', 'edge_attr']
        self.num_pos_feats = embed_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.shape = (1, img_shape, img_shape)
        self.posEncodingDict = self.getPosEncoding()
        self.attrs = attr

    def getPosEncoding(self, mask=None):
        if mask is None:
            mask = torch.zeros(self.shape, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 输出的shape和not_mask shape一致
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # 每一个维度的最后一个值归一
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (
                2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(1, 2, 3, 0). \
            squeeze(-1)# h,w,n,b
        return pos

    def __call__(self, data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                num, dim = value.shape
                coord = value - value.min()
                coord = coord / coord.max() * (self.shape[1] - 1)
                coord = coord.view(-1, 2).to(torch.long)

                posEnCoord = self.posEncodingDict[coord[:, 1], coord[:, 0], :] # XY和HW coord需要对齐
                posEnCoord = posEnCoord.view(num, -1).to(torch.float)
                store[key] = posEnCoord
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@functional_transform('adaptive_translation')
class AdaptiveTranslation(BaseTransform):
    """
    由于生成图像时，有padding的过程，如果需要将户型图的端点坐标和其在图像中的坐标对应，
    需要将端点坐标进行一定的缩放和平移
    在这一步，节点属性->store['x'], 以及边属性->store['edge_attr']中关于坐标的属性
    全部被缩放到[0, img_height(width)区间]
    """
    def __init__(self, img_shape=512, attr=None):
        self.img_shape = img_shape
        if attr is None:
            attr = ['x', 'edge_attr']
        self.attrs = attr

    def __call__(self, data):
        # data.stores['x']中既包含了三角剖分的内点，又包含了三角剖分的顶点，
        # 所以XY轴的最大值表示了户型图最大范围
        for store in data.stores:
            if not store.__contains__('x'):
                raise RuntimeError('there should be property x in data element.')
            nodeCoords = store['x']
            numNode, nodeDim = nodeCoords.shape
            nodeCoords = nodeCoords.view(numNode, -1, 2)
            nodeCoordsX, nodeCoordsY = nodeCoords[:, :, 0], nodeCoords[:, :, 1]

            wMax, hMax = torch.max(nodeCoordsX).item(), torch.max(nodeCoordsY).item() # XY == WH
            if wMax >= hMax:
                scaleFactor = wMax / (self.img_shape - 1) # 最大的坐标是 img_shape - 1
                hRe = int(hMax // scaleFactor)
                wRe = self.img_shape
                if hRe % 2 == 1:
                    hRe = hRe + 1
            else:
                scaleFactor = hMax / (self.img_shape - 1)
                wRe = int(wMax // scaleFactor)
                hRe = self.img_shape
                if wRe % 2 == 1:
                    wRe = wRe + 1
            wPad = int((self.img_shape - 1 - wRe) / 2) #
            hPad = int((self.img_shape - 1 - hRe) / 2) #
            pad = torch.empty((1, 2), dtype=torch.float)
            pad[0, 0] = wPad
            pad[0, 1] = hPad

            for key, value in store.items(*self.attrs):
                value = value / scaleFactor
                stackNum = value.shape[-1] // 2

                if stackNum > 1:
                    stackTs = [pad] * stackNum
                    addedPad = torch.cat(stackTs, dim=-1)
                else:
                    addedPad = pad

                value = value + addedPad
                store[key] = value

        return data


@functional_transform('embd_edge_type')
class EdgeTypeEmbedding(BaseTransform):
    """
    目前只针对fplanVenoGraph的操作,将边的特征根据是否是墙体,
    转化成nn.Embedding()
    """
    def __init__(self, embed_dim=128):
        self.edge_type_embed = nn.Embedding(
            2, embed_dim
        )
        self.embed_dim = embed_dim

    def __call__(self, data: Any) -> Any:
        for store in data.stores:
            if not store.__contains__('edge_type'):
                raise RuntimeError('there should be property edge type in data element.')
            edgeType = torch.squeeze(store['edge_type']).to(torch.long)
            edgeTypeEbd = self.edge_type_embed(edgeType).detach()
            store['edge_type'] = edgeTypeEbd
            # if not store.__contains__('edge_attr'):
            #     raise RuntimeError('there should be property edge attr in data element.')
            # edgeAttr = store['edge_attr']
            # half = edgeAttr.shape[1] // 2
            # fHalf, sHalf = edgeAttr[:, :half], edgeAttr[:, half:]
            # finalHalf = fHalf + sHalf
            #
            # store['edge_attr'] = torch.cat([edgeTypeEbd, finalHalf], dim=1)

        return data


@functional_transform('one_hot_edge_type')
class EdgeTypeOneHot(BaseTransform):
    """
    同EdgeAttrProc.区别在于，使用one-hot向量来做边特征
    """
    def __init__(self, proc_name='edge_type'):
        self.type_embed = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
        self.proc_name = proc_name

    def __call__(self, data: Any) -> Any:
        for store in data.stores:
            if not store.__contains__(self.proc_name):
                raise RuntimeError('there should be property edge type in data element.')
            edgeType = torch.squeeze(store[self.proc_name]).to(torch.long)
            edgeType[edgeType > 1] = 0
            edgeTypeEbd = self.type_embed[edgeType]
            store[self.proc_name] = edgeTypeEbd

        return data


@functional_transform('my_to_undirected_1')
class MyToUndirected(BaseTransform):
    def __init__(self, proc_name=None):
        # 处理edge_index之外的其它index
        if proc_name is None:
            proc_name = ['edge_index_d2', 'edge_index_d4', 'edge_index_d8']
        self.proc_name = proc_name

    def __call__(self, data):
        for store in data.stores:
            if isinstance(self.proc_name, List):
                for n in self.proc_name:
                    store[n] = to_undirected(store[n])
            elif isinstance(self.proc_name, str):
                n = self.proc_name
                store[n] = to_undirected(store[n])

        return data


EdgeAttrProc1 = EdgeTypeOneHot
EdgeAttrProc = EdgeTypeEmbedding