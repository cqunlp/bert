# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Pool class for Encoder"""
from typing import Optional

import mindspore
from mindspore import nn
from mindspore import ops


class MaxPool(nn.Cell):
    """
    Max-pooling Module.

    Args:
        kernel_size (int, Optional): size of max pooling. Default: tensor.shape[-1].
        stride (int): The stride of max pooling. Default: 1.
        dimension: dimension of MaxPool, supported dimension [1,2]. Default: 1.
        pad_mode: 1.same 2.valid , default "valid"

    """

    def __init__(self, kernel_size: Optional[int] = None, stride: int = 1, dimension: int = 1, pad_mode: str = "valid"):
        super(MaxPool, self).__init__()
        if dimension not in [1, 2]:
            raise AssertionError('Now we only support 1d or 2d Pooling')
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply MaxPool

        Args:
            x: (Tensor): The shape is (N, L, C). The input tensor.

        Returns:
            x: (Tensor): The shape is (N, C). The output tensor.
        """

        if self.dimension == 1:
            x = x.transpose((0, 2, 1))  # [N,L,C] -> [N,C,L]
            pooling = nn.MaxPool1d(
                kernel_size=self.kernel_size if self.kernel_size is not None else x.shape[-1],
                stride=self.stride, pad_mode=self.pad_mode)
        else:
            pooling = nn.MaxPool2d(
                kernel_size=self.kernel_size if self.kernel_size is not None else (x.shape[-2], x.shape[-1]),
                stride=self.stride, pad_mode=self.pad_mode)
        x = pooling(x)
        return x.squeeze(axis=-1)


class MaxPoolWithMask(nn.Cell):
    """
    Max pooling with mask, while max-pooling without considering zero in the mask.
    """

    def __init__(self):
        super(MaxPoolWithMask, self).__init__()
        self.inf = 10e12

    def construct(self, tensor: mindspore.Tensor, mask: mindspore.Tensor, axis: int = 1) -> mindspore.Tensor:
        """
        Apply MaxPoolWithMask.

        inputs:
            tensor (Tensor): The shape is (batch_size, seq_len, channels). The input tensor.
            mask (Tensor): The shape is (batch_size, seq_len). The mask tensor. It's value should be 0/1 or True/False.
            axis (int): The dimension when max pooling. Default: 1.

        outputs:
            tensor, the output tensor after max-pooling with mask.
        """

        masks = mask.view(mask.shape[0], mask.shape[1], -1)
        shape = (-1, -1, tensor.shape[-1])
        broadcast_to = ops.BroadcastTo(shape)
        masks = broadcast_to(masks).astype(mindspore.float32)
        masks = (masks <= 0.5).astype(mindspore.float32)
        return (ops.ArgMaxWithValue(axis=axis)(tensor + masks * -self.inf))[1]


class KMaxPool(nn.Cell):
    """
    K max-pooling module.

    Args:
        k (int): The k value of KMaxPool.

    """

    def __init__(self, k: int = 1):
        super(KMaxPool, self).__init__()
        self.k = k

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply KMaxPool.

        inputs:
            x : (Tensor): The shape is (N, L, C). The input tensor.

        outputs:
            x : (Tensor): The shape is (N, C*k). The result of k-max pool.
        """
        x = x.transpose((0, 2, 1))  # [N, L, C] -> [N, C, L]
        topk = ops.TopK()
        x = topk(x, self.k)[0]
        x = x.reshape((x.shape[0], -1))
        return x


class AvgPool(nn.Cell):
    """
    Avg pooling at the last dimension.
    """

    def __init__(self, stride: int = 1, dimension: int = 1, pad_mode: str = "valid"):
        super(AvgPool, self).__init__()
        self.stride = stride
        self.dimension = dimension
        self.pad_mode = pad_mode

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply AvgPool.

        inputs:
            x (Tensor): The shape is (N, L, C). The input tensor.

        outputs:
            x (Tensor): The shape is (N, C). The result of avg pool.
        """
        # [N,L,C] -> [N,C,L]
        if self.dimension == 1:
            x = x.transpose((0, 2, 1))
            kernel_size = x.shape[-1]
            pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=self.stride, pad_mode=self.pad_mode)
        else:
            kernel_size = (x.shape[-2], x.shape[-1])
            pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=self.stride, pad_mode=self.pad_mode)
        x = pooling(x)
        return x.squeeze(axis=-1)


class AvgPoolWithMask(nn.Cell):
    """
    Avg pooling at the last dimension with mask tensor.
    """

    def __init__(self):
        super(AvgPoolWithMask, self).__init__()
        self.inf = 10e12

    def construct(self, tensor: mindspore.Tensor, mask: mindspore.Tensor, axis: int = 1) -> mindspore.Tensor:
        """
        Apply AvgPoolWithMask.

        inputs:
            tensor : (Tensor): (batch_size, seq_len, channels). The input tensor.
            mask : (Tensor): (batch_size, seq_len). The mask tensor. It's value should be 0/1 or True/False.
            axis : (int): The dimension when max pooling. Default: 1.

        outputs:
            tensor : after AvgPooling with mask
        """

        masks = mask.view(mask.shape[0], mask.shape[1], -1).astype(mindspore.float32)
        reducesum = ops.ReduceSum()
        return reducesum(tensor * masks, axis) / reducesum(masks, axis)
