# code from https://github.com/pytorch/opacus/blob/main/opacus/utils/tensor_utils.py

"""
Utils for generating stats from torch tensors.
"""
import math
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_sample_norms(
    named_params: Iterator[Tuple[str, torch.Tensor]], *, flat: bool = True
) -> List[torch.Tensor]:
    r"""
    Calculates the norm of the given tensors for each sample.

    This function calculates the overall norm of the given tensors for each sample,
    assuming each batch's dim is zero.

    Args:
        named_params: An iterator of tuples <name, param> with name being a
            string and param being a tensor of shape ``[B, ...]`` where ``B``
            is the size of the batch and is the 0th dimension.
        flat: A flag, when set to `True` returns a flat norm over all
            layers norms

    Returns:
        A list of tensor norms where length of the list is the number of layers

    Example:
        >>> t1 = torch.rand((2, 5))
        >>> t2 = torch.rand((2, 5))
        >>> norms = calc_sample_norms([("1", t1), ("2", t2)])
        >>> norms, norms[0].shape
        ([tensor([...])], torch.Size([2]))
    """
    norms = [param.view(len(param), -1).norm(2, dim=-1) for name, param in named_params]
    # calc norm over all layer norms if flat = True
    if flat:
        norms = [torch.stack(norms, dim=0).norm(2, dim=0)]
    return norms


def calc_sample_norms_one_layer(param: torch.Tensor) -> torch.Tensor:
    r"""
    Calculates the norm of the given tensor (a single parameter) for each sample.

    This function calculates the overall norm of the given tensor for each sample,
    assuming each batch's dim is zero.

    It is equivalent to:
    `calc_sample_norms(named_params=((None, param),))[0]`

    Args:
        param: A tensor of shape ``[B, ...]`` where ``B``
            is the size of the batch and is the 0th dimension.

    Returns:
        A tensor of norms

    Example:
        >>> t1 = torch.rand((2, 5))
        >>> norms = calc_sample_norms_one_layer(t1)
        >>> norms, norms.shape
        (tensor([...]), torch.Size([2]))
    """
    norms = param.view(len(param), -1).norm(2, dim=-1)
    return norms


def sum_over_all_but_batch_and_last_n(
    tensor: torch.Tensor, n_dims: int
) -> torch.Tensor:
    r"""
    Calculates the sum over all dimensions, except the first
    (batch dimension), and excluding the last n_dims.

    This function will ignore the first dimension, and it will
    not aggregate over the last n_dims dimensions.

    Args:
        tensor: An input tensor of shape ``(B, ..., X[n_dims-1])``.
        n_dims: Number of dimensions to keep.

    Returns:
        A tensor of shape ``(B, ..., X[n_dims-1])``

    Example:
        >>> tensor = torch.ones(1, 2, 3, 4, 5)
        >>> sum_over_all_but_batch_and_last_n(tensor, n_dims=2).shape
        torch.Size([1, 4, 5])
    """
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)


def unfold2d(
    input,
    *,
    kernel_size: Tuple[int, int],
    padding: Union[str, Tuple[int, int]],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
):
    """
    See :meth:`~torch.nn.functional.unfold`
    """
    *shape, H, W = input.shape
    if padding == "same":
        total_pad_H = dilation[0] * (kernel_size[0] - 1)
        total_pad_W = dilation[1] * (kernel_size[1] - 1)
        pad_H_left = math.floor(total_pad_H / 2)
        pad_H_right = total_pad_H - pad_H_left
        pad_W_left = math.floor(total_pad_W / 2)
        pad_W_right = total_pad_W - pad_W_left

    elif padding == "valid":
        pad_W_left, pad_W_right, pad_H_left, pad_H_right = (0, 0, 0, 0)
    else:
        pad_H_left, pad_H_right, pad_W_left, pad_W_right = (
            padding[0],
            padding[0],
            padding[1],
            padding[1],
        )

    H_effective = (
        H
        + pad_H_left
        + pad_H_right
        - (kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1))
    ) // stride[0] + 1
    W_effective = (
        W
        + pad_W_left
        + pad_W_right
        + -(kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1))
    ) // stride[1] + 1
    # F.pad's first argument is the padding of the *last* dimension
    input = F.pad(input, (pad_W_left, pad_W_right, pad_H_left, pad_H_right))
    *shape_pad, H_pad, W_pad = input.shape
    strides = list(input.stride())
    strides = strides[:-2] + [
        W_pad * dilation[0],
        dilation[1],
        W_pad * stride[0],
        stride[1],
    ]
    out = input.as_strided(
        shape + [kernel_size[0], kernel_size[1], H_effective, W_effective], strides
    )

    return out.reshape(input.size(0), -1, H_effective * W_effective)


def unfold3d(
    tensor: torch.Tensor,
    *,
    kernel_size: Union[int, Tuple[int, int, int]],
    padding: Union[int, Tuple[int, int, int]] = 0,
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
):
    r"""
    Extracts sliding local blocks from an batched input tensor.

    :class:`torch.nn.Unfold` only supports 4D inputs (batched image-like tensors).
    This method implements the same action for 5D inputs

    Args:
        tensor: An input tensor of shape ``(B, C, D, H, W)``.
        kernel_size: the size of the sliding blocks
        padding: implicit zero padding to be added on both sides of input
        stride: the stride of the sliding blocks in the input spatial dimensions
        dilation: the spacing between the kernel points.

    Returns:
        A tensor of shape ``(B, C * np.product(kernel_size), L)``, where L - output spatial dimensions.
        See :class:`torch.nn.Unfold` for more details

    Example:
        >>> B, C, D, H, W = 3, 4, 5, 6, 7
        >>> tensor = torch.arange(1, B*C*D*H*W + 1.).view(B, C, D, H, W)
        >>> unfold3d(tensor, kernel_size=2, padding=0, stride=1).shape
        torch.Size([3, 32, 120])
    """

    if len(tensor.shape) != 5:
        raise ValueError(
            f"Input tensor must be of the shape [B, C, D, H, W]. Got{tensor.shape}"
        )

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if isinstance(padding, int):
        padding = (padding, padding, padding)

    if isinstance(stride, int):
        stride = (stride, stride, stride)

    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    if padding == "same":
        total_pad_D = dilation[0] * (kernel_size[0] - 1)
        total_pad_H = dilation[1] * (kernel_size[1] - 1)
        total_pad_W = dilation[2] * (kernel_size[2] - 1)
        pad_D_left = math.floor(total_pad_D / 2)
        pad_D_right = total_pad_D - pad_D_left
        pad_H_left = math.floor(total_pad_H / 2)
        pad_H_right = total_pad_H - pad_H_left
        pad_W_left = math.floor(total_pad_W / 2)
        pad_W_right = total_pad_W - pad_W_left

    elif padding == "valid":
        pad_D_left, pad_D_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right = (
            0,
            0,
            0,
            0,
            0,
            0,
        )
    else:
        pad_D_left, pad_D_right, pad_H_left, pad_H_right, pad_W_left, pad_W_right = (
            padding[0],
            padding[0],
            padding[1],
            padding[1],
            padding[2],
            padding[2],
        )

    batch_size, channels, _, _, _ = tensor.shape

    # Input shape: (B, C, D, H, W)
    tensor = F.pad(
        tensor,
        (pad_W_left, pad_W_right, pad_H_left, pad_H_right, pad_D_left, pad_D_right),
    )
    # Output shape: (B, C, D+pad_W_left+pad_W_right, H+pad_H_left+pad_H_right, W+pad_D_left+pad_D_right)

    dilated_kernel_size = (
        kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1),
        kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1),
        kernel_size[2] + (kernel_size[2] - 1) * (dilation[2] - 1),
    )

    tensor = tensor.unfold(dimension=2, size=dilated_kernel_size[0], step=stride[0])
    tensor = tensor.unfold(dimension=3, size=dilated_kernel_size[1], step=stride[1])
    tensor = tensor.unfold(dimension=4, size=dilated_kernel_size[2], step=stride[2])

    if dilation != (1, 1, 1):
        tensor = filter_dilated_rows(tensor, dilation, dilated_kernel_size, kernel_size)

    # Output shape: (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
    # For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`

    tensor = tensor.permute(0, 2, 3, 4, 1, 5, 6, 7)
    # Output shape: (B, D_out, H_out, W_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    tensor = tensor.reshape(batch_size, -1, channels * np.prod(kernel_size)).transpose(
        1, 2
    )
    # Output shape: (B, D_out * H_out * W_out, C * kernel_size[0] * kernel_size[1] * kernel_size[2]

    return tensor


def filter_dilated_rows(
    tensor: torch.Tensor,
    dilation: Tuple[int, int, int],
    dilated_kernel_size: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
):
    """
    A helper function that removes extra rows created during the process of
    implementing dilation.

    Args:
        tensor: A tensor containing the output slices resulting from unfolding
                the input tensor to `unfold3d()`.
                Shape is ``(B, C, D_out, H_out, W_out, dilated_kernel_size[0],
                dilated_kernel_size[1], dilated_kernel_size[2])``.
        dilation: The dilation given to `unfold3d()`.
        dilated_kernel_size: The size of the dilated kernel.
        kernel_size: The size of the kernel given to `unfold3d()`.

    Returns:
        A tensor of shape (B, C, D_out, H_out, W_out, kernel_size[0], kernel_size[1], kernel_size[2])
        For D_out, H_out, W_out definitions see :class:`torch.nn.Unfold`.

    Example:
        >>> tensor = torch.zeros([1, 1, 3, 3, 3, 5, 5, 5])
        >>> dilation = (2, 2, 2)
        >>> dilated_kernel_size = (5, 5, 5)
        >>> kernel_size = (3, 3, 3)
        >>> filter_dilated_rows(tensor, dilation, dilated_kernel_size, kernel_size).shape
        torch.Size([1, 1, 3, 3, 3, 3, 3, 3])
    """

    kernel_rank = len(kernel_size)

    indices_to_keep = [
        torch.arange(0, dilated_kernel_size[i], dilation[i], device=tensor.device)
        for i in range(kernel_rank)
    ]

    axis_offset = len(tensor.shape) - kernel_rank

    for dim in range(kernel_rank):
        tensor = torch.index_select(
            tensor, dim=axis_offset + dim, index=indices_to_keep[dim]
        )

    return tensor


# ================================
# Module Wrappers for 2D and 3D Unfold
# ================================

class Unfold2d(nn.Module):
    r"""Extracts sliding local blocks from a batched input tensor.
    
    This module provides the same functionality as :class:`torch.nn.Unfold` but with
    enhanced features like "same" and "valid" padding modes.
    
    Args:
        kernel_size (int or tuple): the size of the sliding blocks
        dilation (int or tuple, optional): a parameter that controls the spacing between 
            the kernel points; also known as the à trous algorithm. Default: (1, 1)
        padding (int, tuple or str, optional): implicit zero padding to be added on 
            both sides of input. Can be int, tuple, "same" or "valid". Default: (0, 0)
        stride (int or tuple, optional): the stride of the sliding blocks in the input 
            spatial dimensions. Default: (1, 1)
            
    * :attr:`kernel_size` controls the size of the sliding block
    * :attr:`dilation` controls the spacing between the kernel points
    * :attr:`padding` controls the amount of implicit zero padding on both sides
    * :attr:`stride` controls the stride for the cross-correlation
    
    Shape:
        - Input: :math:`(B, C, H, W)` where B is the batch size
        - Output: :math:`(B, C \times \prod(\text{kernel\_size}), L)` where L is the total
          number of such blocks
    
    Example::
    
        >>> unfold = Unfold2d(kernel_size=3, padding="same", stride=1)
        >>> input = torch.randn(2, 5, 12, 12)
        >>> output = unfold(input)
        >>> # each patch contains 45 values (5*3*3) and there are 12*12 patches in total
        >>> output.size()
        torch.Size([2, 45, 144])
    """
    
    __constants__ = ['kernel_size', 'dilation', 'padding', 'stride']
    
    def __init__(
        self, 
        kernel_size: Union[int, Tuple[int, int]], 
        dilation: Union[int, Tuple[int, int]] = (1, 1), 
        padding: Union[int, Tuple[int, int], str] = (0, 0), 
        stride: Union[int, Tuple[int, int]] = (1, 1)
    ) -> None:
        super().__init__()
        
        # Normalize parameters to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return unfold2d(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )
        
    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}'.format(**self.__dict__)


class Unfold3d(nn.Module):
    r"""Extracts sliding local blocks from a batched input tensor.
    
    This module is a 3D extension of :class:`torch.nn.Unfold`, supporting 5D inputs
    of shape (B, C, D, H, W).
    
    Args:
        kernel_size (int or tuple): the size of the sliding blocks
        dilation (int or tuple, optional): a parameter that controls the spacing between 
            the kernel points; also known as the à trous algorithm. Default: (1, 1, 1)
        padding (int, tuple or str, optional): implicit zero padding to be added on 
            both sides of input. Can be int, tuple, "same" or "valid". Default: (0, 0, 0)
        stride (int or tuple, optional): the stride of the sliding blocks in the input 
            spatial dimensions. Default: (1, 1, 1)
            
    * :attr:`kernel_size` controls the size of the sliding block
    * :attr:`dilation` controls the spacing between the kernel points
    * :attr:`padding` controls the amount of implicit zero padding on both sides
    * :attr:`stride` controls the stride for the cross-correlation
    
    Shape:
        - Input: :math:`(B, C, D, H, W)` where B is the batch size
        - Output: :math:`(B, C \times \prod(\text{kernel\_size}), L)` where L is the total
          number of such blocks
    
    Example::
    
        >>> unfold = Unfold3d(kernel_size=3, padding="same", stride=1)
        >>> input = torch.randn(2, 5, 10, 12, 12)
        >>> output = unfold(input)
        >>> # each patch contains 135 values (5*3*3*3) and there are 10*12*12 patches in total
        >>> output.size()
        torch.Size([2, 135, 1440])
    """
    
    __constants__ = ['kernel_size', 'dilation', 'padding', 'stride']
    
    def __init__(
        self, 
        kernel_size: Union[int, Tuple[int, int, int]], 
        dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1), 
        padding: Union[int, Tuple[int, int, int], str] = (0, 0, 0), 
        stride: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    ) -> None:
        super().__init__()
        
        # Normalize parameters to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return unfold3d(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )
        
    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, dilation={dilation}, padding={padding}, stride={stride}'.format(**self.__dict__)

