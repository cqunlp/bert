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
"""StaticEmbedding class."""
import os
import warnings
from typing import Union
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from .embedding import TokenEmbedding
from ..common.data.vocabulary import Vocabulary


class StaticEmbedding(TokenEmbedding):
    """
    Given the name or path of the pretrained embedding, corresponding embedding is extracted from the pretrained
    embedding according to vocab. If embedding is not found, we initialize it with a random embedding.

    Args:
        vocab(Vocabulary): Vocabulary object. StaticEmbedding will only load the word vector of the word contained
        in the word list, using a random initialization if it is not found in the pretrained embedding.
        model_dir_or_name(Union[str, None]):There are two ways to call pretrained static embedding: the first way
        is to pass in the local folder (there should be only one file with the suffix .txt) or the file path. The
        second is the name of the passed embedding. In the second case, the embedding will automatically check
        whether the model exists in the cache. If not, the embedding will be automatically downloaded (after
        huawei cloud implementation). If the input is None, an embedding is randomly initialized using the dimension
        embedding_dim.
        embedding_path(Union[str, None]): The path of embedding file.
        embedding_dim(int): The dimension of randomly initialized embedding. model_dir_or_name will be ignored if the
        value is greater than 0.
        requires_grad(bool, Optional): Default：True.
        dropout(float, Optional): the probability of Dropout layer for the representation of the embedding.

    Returns:
        Tensor.

    Examples:
        >>> vocab = Vocabulary()
        >>> vocab.update(["i", "am", "fine"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5)
        >>> words = mindspore.Tensor([[vocab[word] for word in ["i", "am", "fine"]]])
        >>> embed(words)
        >>> Tensor(shape=[1, 3, 5], dtype=Float32, value=
            [[[7.56267071e-001, -3.02625038e-002, 6.10783875e-001, 4.03315663e-001, -6.82987273e-001],
            [3.35875869e-001, 2.93195043e-002, 2.17977986e-001, 9.68403295e-002, -4.01605248e-001],
            [-2.35586300e-001, 4.89649743e-001, -2.10691467e-001, -1.81295246e-001, -6.90823942e-002]]]).
    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: Union[str, None] = None,
                 embedding_path: Union[str, None] = None, embedding_dim=-1, requires_grad: bool = True, dropout=0.1):
        super(StaticEmbedding, self).__init__(vocab, dropout=dropout)
        if embedding_dim > 0 and model_dir_or_name:
            warnings.warn(f"StaticEmbedding will ignore {model_dir_or_name}, and randomly initialize embedding with"
                          f" dimension {embedding_dim}. If you want to use pre-trained embedding, set embedding_dim"
                          f" to 0.")
            embedding_dim = int(embedding_dim)
            model_dir_or_name = None
        model_path = None
        if model_dir_or_name:
            model_path = embedding_path
        if model_path:
            embedding = self._load_with_vocab(model_path, vocab)
        else:
            embedding = self._randomly_init_embed(len(vocab), embedding_dim)
        embedding_weight = mindspore.Tensor(embedding, dtype=mindspore.float32)
        self.embedding = nn.Embedding(vocab_size=embedding_weight.shape[0],
                                      embedding_size=embedding_weight.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      embedding_table=embedding_weight)
        self._embed_size = self.embedding.embedding_size
        self.requires_grad = requires_grad

    def construct(self, words):
        words = self.embedding(words)
        words = self.dropout(words)
        return words

    def _randomly_init_embed(self, num_embedding: int, embedding_dim: int) -> np.ndarray:
        random_vector = np.random.uniform(-np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim),
                                          (num_embedding, embedding_dim))
        return random_vector

    def _load_with_vocab(self, embed_filepath: str, vocab: Vocabulary, padding='<pad>',
                         unknown='<unk>') -> np.ndarray:
        """

        Args:
            embed_filepath(str): The path of pretrained embedding.
            vocab(Vocabulary): Read the embedding of words that appear in vocab. if a word that does not appear in
            vocab, it will be sampled by the normal distribution so that the whole embedding is uniformly distributed.
            padding(str): The padding token in vocabulary.
            unknown(str): The unknown token in vocabulary.

        Returns:
            numpy.ndarray.
        """
        if not isinstance(vocab, Vocabulary):
            raise AssertionError("Only Vocabulary class is supported.")
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError(f"{embed_filepath} does not exist.")
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.padding_idx] = ops.Zeros()(dim, mindspore.float32)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = ops.Zeros()(dim, mindspore.float32)
            found_unknown = False
            for _, line in enumerate(f, start_idx):
                parts = line.strip().split()
                word = ''.join(parts[:-dim])
                nums = parts[-dim:]
                if word == padding and vocab.padding:
                    word = vocab.padding
                elif word == unknown and vocab.unknown:
                    word = vocab.unknown
                    found_unknown = True
                if word in vocab.word_count.keys():
                    index = vocab[word]
                    matrix[index] = np.fromstring(' '.join(nums), sep=' ', count=dim)
            for word in vocab.word_count.keys():
                index = vocab[word]
                if index not in matrix.keys():
                    if found_unknown:
                        matrix[index] = matrix[vocab.unknown_idx]
                    else:
                        matrix[index] = None
            vectors = self._randomly_init_embed(len(matrix), dim)
            if not vocab.unknown:
                vocab.unknown_idx = len(matrix)
                vectors = ops.Concat()(vectors, ops.Zeros()(1, dim))
            index = 0
            for word in vocab.word_count.keys():
                index_in_vocab = vocab[word]
                if index_in_vocab in matrix.keys():
                    vec = matrix.get(index_in_vocab)
                    if vec is not None:
                        vectors[index] = vec
                        index += 1
            return vectors
