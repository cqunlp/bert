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
"""Embedding class for Transformer."""

import math
import numpy as np

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer as init
from mindspore.common.initializer import Normal


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 use_one_hot_embeddings: bool = False):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(mindspore.Tensor(init(Normal(), [vocab_size, embedding_size])))
        self.shape_flat = (-1,)
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = mindspore.Tensor(1.0, mstype.float32)
        self.off_value = mindspore.Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()

    def construct(self, input_ids: mindspore.Tensor) -> (mindspore.Tensor, mindspore.Tensor):
        """
        Get a embeddings lookup table with a fixed dictionary and size.

        Args:
            input_ids (Tensor): The id tokens of input sequence.

        Returns:
            output (Tuple[Tensor, Tensor]): The word embedding and the embedding lookup table of input sequence.
        """
        input_shape = input_ids.shape

        flat_ids = input_ids.reshape(self.shape_flat)
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_size,)
        output = output_for_reshape.reshape(out_shape)

        return output, self.embedding_table


def position_encoding(length: int,
                      hidden_size: int,
                      min_timescale: int = 1,
                      max_timescale: float = 1e4) -> mindspore.Tensor:
    """
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        hidden_size (int): Hidden size.
        min_timescale (float): Default: 1.
        max_timescale (float): Default: 10000.

    Returns:
        Tensor of shape (length, hidden_size)
    """
    hidden_size = hidden_size // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (hidden_size - 1))
    inv_timescales = min_timescale * np.exp(np.arange(hidden_size, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_size (int): The size of each embedding vector.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        dropout_prob (float): The dropout probability. Default: 0.1.
    """

    def __init__(self,
                 embedding_size: int,
                 max_position_embeddings: int = 128,
                 dropout_prob: float = 0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.scores_mul = mindspore.Tensor([math.sqrt(float(embedding_size))], dtype=mstype.float32)
        self.multiply = P.Mul()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=mstype.float32)
        self.use_dropout = dropout_prob > 0
        self.expand_dims = P.ExpandDims()
        self.position_embedding_table = mindspore.Tensor(position_encoding(max_position_embeddings, embedding_size),
                                                         mstype.float32)

    def construct(self, word_embeddings: mindspore.Tensor) -> mindspore.Tensor:
        """
        Postprocessors apply positional embeddings to word embeddings.

        Args:
            word_embeddings (Tensor): The word_embedding of input sequence.

        Returns:
            output (Tensor): The result of add position embeddings to word embeddings.
        """
        input_shape = word_embeddings.shape
        input_len = input_shape[1]

        output = self.multiply(word_embeddings, self.scores_mul)

        # add position embeddings
        position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(output, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output
