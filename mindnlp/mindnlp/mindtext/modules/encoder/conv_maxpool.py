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
"""
ConvMaxpool class.
A layer that combines a single Convolution and a Max-Pooling.
Given an input (batch_size x max_len x input_size), return a tensor (batch_size x sum(output_channels)).
First, CNN is used to conduct convolution for the input, then the output of CNN is passed through the
activation layer, and finally max_pooling is performed in the dimension of max_len.
"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.layer.activation import _activation, get_activation


class ConvMaxpool(nn.Cell):
    """
    A layer that combines a single Convolution and a Max-Pooling.

    Args:
        in_channels (int): The size of input channel, generally, embedding size.
        out_channels (int): The size of output channel.
        kernel_size (int): The size of kernel.
        stride (int): Stride length.
        padding (int): Padding value.
        has_bias (bool): if bias.
        activation (str): activation function (relu, sigmoid, tanh ...).

    Returns:
        Tensor.

    Examples
        conv_maxpool = ConvMaxpool(in_channels=128, out_channels=3, kernel_sizes=3)
        example = Tensor(np.random.rand(2, 128, 128), mindspore.float32)
        output = conv_maxpool(example)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=False, activation="relu"):
        super(ConvMaxpool, self).__init__()
        if not kernel_size % 2 == 1:
            raise AssertionError("kernel size has to be odd numbers.")
        # convolution
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, has_bias=has_bias)
        # activation function
        if activation in _activation.keys():
            self.activation = get_activation(activation)
        else:
            raise Exception(
                "Undefined activation function")

    def construct(self, x):
        """
        Args:
            Tensor x: batch_size x max_len x input_size

        Returns:
            Tensor.
        """
        x = ops.Transpose()(x, (0, 2, 1))
        # convolution
        conv = self.conv(x)
        x = self.activation(conv)
        # max-pooling
        max_pool = nn.MaxPool1d(kernel_size=x.shape[2])
        x = max_pool(x).squeeze(2)
        return x
