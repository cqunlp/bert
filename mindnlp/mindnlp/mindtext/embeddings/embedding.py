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
"""Embedding class."""
from typing import Union, Tuple
from abc import abstractmethod
import numpy as np

import mindspore
import mindspore.nn as nn
from .utils import get_embeddings
from ..common.data.vocabulary import Vocabulary


class Embedding(nn.Cell):
    """
    Word embedding, which supports input initialization in a variety of ways. You can get the word table size
    with self.num_embeddings and the dimension of the embedding with self.embedding_dim.

    Args:
        init_embed (tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray): The Embedding size
        (input tuple(int, int), the fist one is vocab_size, and the second is embedding_dim); or input
        Tensor, Embedding, numpy.ndarray to initialieze embedding.
        dropout(float): the probability of Dropout layer for the representation of the embedding.

    Returns:
        Tensor.

    Examples:
        >>> import numpy as np
        >>> init_embed = (200, 100)
        >>> embedding = Embedding(init_embed, 0.1)
        >>> init_embed = np.random.randn(4, 3)
        >>> embedding = Embedding(init_embed, 0.1)

    """

    def __init__(self, init_embed: Union[Tuple[int, int], np.ndarray, mindspore.Tensor, nn.Cell], dropout: float = 0.1):
        super(Embedding, self).__init__()
        self.embeddings = get_embeddings(init_embed)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self._embed_size = self.embeddings.embedding_size

    def __len__(self) -> int:
        return len(self.embeddings)

    def construct(self, words: mindspore.Tensor) -> mindspore.Tensor:
        embed = self.embeddings(words)
        return self.dropout(embed)

    @property
    def embed_size(self) -> int:
        return self._embed_size

    @property
    def embedding_dim(self) -> int:
        return self._embed_size

    @property
    def embedding_size(self) -> int:
        return self._embed_size

    @property
    def num_embeddings(self) -> int:
        return self.embeddings.vocab_size


class TokenEmbedding(nn.Cell):
    """Base classes for all Embedding class."""

    def __init__(self, vocab: Vocabulary, dropout: float = 0.1):
        super(TokenEmbedding, self).__init__()
        vocab.build_vocab()
        self._word_vocab = vocab
        self._word_pad_index = vocab.padding_idx
        self._word_unk_index = vocab.unknown_idx
        self.dropout_layer = nn.Dropout(keep_prob=1.0 - dropout)

    def __len__(self) -> int:
        return len(self._word_vocab)

    def get_word_vocab(self) -> Vocabulary:
        return self._word_vocab

    def dropout(self, words: mindspore.Tensor) -> mindspore.Tensor:
        return self.dropout_layer(words)

    @property
    def embed_size(self) -> int:
        return self.embed_size

    @property
    def embedding_dim(self) -> int:
        return self.embed_size

    @property
    def embedding_size(self) -> int:
        return self._embed_size

    @property
    def num_embeddings(self) -> int:
        return len(self._word_vocab)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.num_embeddings, self.embedding_dim

    @abstractmethod
    def construct(self, words: mindspore.Tensor):
        raise NotImplementedError
