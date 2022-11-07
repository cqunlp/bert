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
"""Normal decoder class."""
import mindspore
import mindspore.nn as nn


class NormalDecoder(nn.Cell):
    """
    The normal decoder, containing a dropout layer and a dense layer.

    Args:
        num_filters(int): The number of filters. Default: 256.
        num_classes(int): The number of output classes.
        classes_dropout(float): The probability of Dropout layer. Default: 0.1.

    Returns:
        Tensor.

    Examples:

    """

    def __init__(self, num_filters: int, num_classes: int, classes_dropout: float = 0.1, activation: str = 'relu'):
        super(NormalDecoder, self).__init__()
        self.drop_out = nn.Dropout(keep_prob=1 - classes_dropout)
        self.linear = nn.Dense(num_filters, num_classes, activation=activation)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Apply NormalDecoder."""
        x = self.drop_out(x)
        x = self.linear(x)
        return x
