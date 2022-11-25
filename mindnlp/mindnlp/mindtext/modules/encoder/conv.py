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
"""Convolution encoder class."""
import mindspore.nn as nn
import mindspore.ops as ops
from mindtext.embeddings.region_embedding import RegionEmbedding
from mindtext.embeddings.static_embedding import StaticEmbedding


class ConvEncoder(nn.Cell):
    """
    The convolution encoder.

    Args:
        init_embed(StaticEmbedding): StaticEmbedding object.
        num_filters(int): The number of filters. Default: 256.
        kernel_size(int): The size of kernel. Default: 3.
        num_layers(int): The number of CNN compositions. Default: 7.
        embed_dropout(float): The probability of Dropout layer. Default: 0.1.

    Returns:
        Tensor. The output of Convolution encoder.

    Examples:
        >>> vocab = Vocabulary()
        >>> vocab.update(["i", "am", "fine"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=100)
        >>> conv_encoder = ConvEncoder(embed)
        >>> words = mindspore.Tensor(np.random.randint(0, 3, (1, 256)))
        >>> x = conv_encoder(words)
    """

    def __init__(self, init_embed: StaticEmbedding, num_filters: int = 256, kernel_size: int = 3,
                 num_layers: int = 7, embed_dropout: float = 0.1):
        super(ConvEncoder, self).__init__()
        self.region_embed = RegionEmbedding(init_embed, out_dim=num_filters, kernel_sizes=[1, 3, 5])
        self.conv_list = nn.CellList()
        for _ in range(num_layers):
            self.conv_list.append(nn.SequentialCell(
                nn.ReLU(),
                nn.Conv1d(num_filters, num_filters, kernel_size),
                nn.Conv1d(num_filters, num_filters, kernel_size)
            ))
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.embed_drop = nn.Dropout(1-embed_dropout)

    def construct(self, words):
        x = self.region_embed(words)
        x = self.embed_drop(x)
        x = self.conv_list[0](x) + x
        for conv in self.conv_list[1:]:
            x = self.pool(x)
            x = conv(x) + x
        _, x = ops.ArgMaxWithValue(axis=-1)(x)
        return x
