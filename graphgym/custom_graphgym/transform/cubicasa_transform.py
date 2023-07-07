import math
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

@functional_transform('sine_positional_encoding')
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
        self.attr = attr

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
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # b,n,h,w
        return pos

    def __call__(self, data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                num, dim = value.shape
                coord = value - value.min()
                coord = coord / coord.max() * (self.shape[1] - 1)
                coord = coord.view(-1, 2).to(torch.int)
                posEnCoord = self.posEncodingDict[:, :, coord]
                posEnCoord = posEnCoord.view(num, -1).to(torch.float)
                value = value
                store[key] = posEnCoord
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
