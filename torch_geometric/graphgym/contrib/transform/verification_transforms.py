import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as F

from typing import Any
from torchvision.transforms import Compose
from torch_geometric.transforms import BaseTransform


class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pReal):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if pReal < self.p:
            return F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip:
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, pReal):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if pReal < self.p:
            return F.vflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class GraphRotate:
    """
    关于attr的说明: 图数据的缩放和旋转并不影响节点在空间中的具体位置，因此
    store['pos']这个属性可以不变换.
    store['x']和store['edge_attr']中的属性需要变换，乘上对应对应的变换
    系数，进行旋转变换

    注意：在不使用图像特征和图节点、边特征对应时，该函数意义不大。
    """
    def __init__(self, rotation=False, rotationAnchor=255, rotationDegree=(-90, 90), attr=None):
        if attr is None:
            attr = ['x', 'edge_attr']
        self.attrs = attr
        self.rotation = rotation
        self.rotationAnchor = rotationAnchor
        self.rotationDegree = rotationDegree

    def __call__(self, data: Any) -> Any:
        pass


class GraphRandomFlip:
    def __init__(self, p=0.5, attr=None, flip_mode='horizontal'):
        """
        放在transform的最开始
        :param p:
        :param attr:
        :param flip_mode:
        """
        super().__init__()
        self.p = p
        if attr is None:
            attr = ['x', 'edge_attr']
        self.attrs = attr

        # horizontal -> x flip -> dim 0
        # vertical -> y flip -> dim 1
        self.flipDim = 0 if flip_mode == 'horizontal' else 1

    def __call__(self, data: Any, pReal=None) -> Any:
        if pReal is None:
            pReal = np.random.rand(1)

        if pReal < self.p:
            for store in data.stores:
                if not store.__contains__('x'):
                    raise RuntimeError(
                        'In {}. ' 
                        'There should be property x in data element.'.\
                        format(self.__repr__())
                    )
                nodeCoords = store['x'].view(-1, 2)
                flipMax = torch.max(nodeCoords[:, self.flipDim])

                for key, value in store.items(*self.attrs):
                    num, dim = value.shape
                    v = value.view(-1, 2)
                    subT = v[:, self.flipDim]
                    v[:, self.flipDim] = flipMax - subT

                    store[key] = v.view(num, dim)

        return data


def dataAugWrapper(p, aug_mode=None):
    """
    数据集的数据增广操作,目前只实现了flip
    :return:
    """
    # 由于应用该transform的数据集不参与训练，因此不需要将label和image
    # 同时transform
    if aug_mode is None:
        aug_mode = dict(
            flip='horizontal',
            scale=1.0,
            rotation=False,
        )

    if aug_mode['flip'] == 'horizontal':
        imgFlip = RandomHorizontalFlip(p=p)
        graphFlip = GraphRandomFlip(p=p, flip_mode='horizontal')
    else:
        imgFlip = RandomVerticalFlip(p=p)
        graphFlip = GraphRandomFlip(p=p, flip_mode='vertical')

    return imgFlip, graphFlip
