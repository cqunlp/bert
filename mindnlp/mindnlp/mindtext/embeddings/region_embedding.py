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
"""RegionEmbedding class."""
from typing import List, Optional

import mindspore.nn as nn
import mindspore.ops as ops
from ..embeddings.static_embedding import StaticEmbedding


class RegionEmbedding(nn.Cell):
    """
    RegionEmbedding for DPCNN based on the paper `Semi-supervised Convolutional Neural Networks for Text Categorization
    via Region Embedding <https://arxiv.org/abs/1504.01255>`.
    Args:
        init_embed (StaticEmbedding): The embedding needed to operate.
        out_dim (int): The output size of embedding.
        kernel_sizes (list[int]): The sizes of kernel_sizes.
    Returns:
        Tensor.
    Examples:
        >>> from mindtext.common.data.vocabulary import Vocabulary
        >>> vocab = Vocabulary()
        >>> vocab.update(["i", "am", "fine"])
        >>> init_embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=300)
        >>> region_embed = RegionEmbedding(init_embed, 256, [1, 3, 5])
    """

    def __init__(self, init_embed: StaticEmbedding, out_dim=300, kernel_sizes: Optional[List[int]] = None):
        super().__init__()
        self.embed = init_embed
        self.embedding_dim = self.embed.embedding_size
        if not kernel_sizes:
            raise AssertionError('kernel_sizes should not be None.')
        if not isinstance(kernel_sizes, list):
            raise AssertionError('kernel_sizes should be List[int].')
        self.region_embeds = nn.CellList(
            [nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size) for kernel_size in kernel_sizes])
        self.linears = nn.CellList(
            [nn.Conv1d(self.embedding_dim, out_dim, 1) for _ in range(len(kernel_sizes))])

    def construct(self, x):
        x = self.embed(x)
        x = ops.Transpose()(x, (0, 2, 1))
        output = 0
        for conv, linear in zip(self.region_embeds, self.linears[1:]):
            x = conv(x)
            output = output + linear(x)
        return output
