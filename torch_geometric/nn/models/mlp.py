from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity, LayerNorm, InstanceNorm1d

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import NoneType


class MLP(torch.nn.Module):
    r"""A Multi-Layer Perception (MLP) model.
    There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit channel sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128])

       creates a three-layer MLP with **differently** sized hidden layers.

    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        batch_norm_kwargs (Dict[str, Any], optional): Arguments passed to
            :class:`torch.nn.BatchNorm1d` in case :obj:`batch_norm == True`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the module will not
            learn additive biases. (default: :obj:`True`)
        relu_first (bool, optional): Deprecated in favor of :obj:`act_first`.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: str = "relu",
        batch_norm: str = "batch",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        batch_norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: bool = True,
        relu_first: bool = False,
    ):
        super().__init__()

        act_first = act_first or relu_first  # Backward compatibility.
        batch_norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if batch_norm == 'batch':
                norm = BatchNorm1d(hidden_channels, **batch_norm_kwargs)
            elif batch_norm == 'layer':
                norm = LayerNorm(hidden_channels, **batch_norm_kwargs)
            elif batch_norm == 'instance':
                norm = InstanceNorm1d(hidden_channels, **batch_norm_kwargs)
            else:
                norm = Identity()
            self.norms.append(norm)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor, return_emb: NoneType = None) -> Tensor:
        """"""
        for lin, norm in zip(self.lins, self.norms):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = x

        if self.plain_last:
            x = self.lins[-1](x)

        return (x, emb) if isinstance(return_emb, bool) else x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'
