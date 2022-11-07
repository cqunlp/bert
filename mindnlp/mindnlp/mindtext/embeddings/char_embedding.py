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
This file mainly contains character Embedding, including character Embedding based on CNN and LSTM. Like other
Embedding, the Embedding input here is also the index of the word, instead of using the index of the char in the word
to get the expression.
"""
from typing import List, Optional
import numpy as np

import mindspore
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore.nn.layer.activation import _activation, get_activation
from tqdm import tqdm

from ..common.data.vocabulary import Vocabulary
from .embedding import TokenEmbedding
from .static_embedding import StaticEmbedding
from .utils import _construct_char_vocab_from_vocab


class CNNCharEmbedding(TokenEmbedding):
    """
    Use CNN to generate character embedding. The structure of CNN is embed(x) -> Dropout(x) -> CNN(x) -> activation(x)
    -> pool -> fc -> Dropout.The filters of different kernel sizes result in concatenating and passing through a fully
    connected layer, and then outputting the representation of the word.

    Example::

        >>> import mindspore
        >>> from mindtext.common.data.vocabulary import Vocabulary
        >>> from mindtext.embeddings.char_embedding import CNNCharEmbedding
        >>> vocab = Vocabulary()
        >>> vocab.update("The whether is good .".split())
        >>> vocab.build_vocab()
        >>> embed = CNNCharEmbedding(vocab, embed_size=50)
        >>> words = mindspore.Tensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.shape
        >>> # (1, 5, 50)

    Args:
        vocab (Vocabulary): The vocabulary for CNNCharEmbedding.
        embed_size (int): The output dimension of the CNNCharEmbedding. Default: 50.
        char_emb_size (int): The embedding dimension of the character. The character is generated from vocab.
                            Default: 50.
        dropout (float): The dropout rate of the CNNCharEmbedding. Default: 0.1.
        filter_nums (List[int]): The number of filters. The length needs to be consistent with the kernels.
                                Default: [40, 30, 20].
        kernel_sizes (List[int]): The size of kernel. Default: [5, 3, 1].
        pool_method (str): The pool method used when synthesizing the representation of the character into a
                            representation, supports 'avg' and 'max'. Default: max.
        activation (str): The activation method used after CNN, supports 'relu','sigmoid','tanh' or custom functions.
        min_char_freq (int): The minimum frequency of occurrence of a character. Default: 2.
        pre_train_char_embed (str): There are two ways to call the pre-trained character embedding: the first is to pass
                                    in the embedding folder (there should be only one file with .txt as the suffix) or
                                    the file path. The second is to pass in the name of the embedding. In the second
                                    case, it will automatically check whether the model exists in the cache, if not,
                                    it will be downloaded automatically. If the input is None, use the dimension of
                                    embedding_dim to randomly initialize an embedding.
        requires_grad (bool): Whether to update the weight.
        include_word_start_end (bool): Whether to add special marking symbols before and ending the character at the
                                        beginning and end of each word.

    """
    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50,
                 dropout: float = 0.1, filter_nums: List[int] = (30,), kernel_sizes: List[int] = (3,),
                 pool_method: str = 'max', activation: str = 'relu', min_char_freq: int = 1,
                 pre_train_char_embed: Optional[str] = None, requires_grad: bool = True,
                 include_word_start_end: bool = True):

        super(CNNCharEmbedding, self).__init__(vocab, dropout=dropout)
        self.char_emb_size = char_emb_size
        for kernel in kernel_sizes:
            assert kernel % 2 == 1, "Only odd kernel is allowed."

        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method

        # Activation function.
        if activation in _activation.keys():
            self.activation = get_activation(activation)
        else:
            raise Exception(
                "Undefined activation function")

        # Build the vocabulary of chars.
        self.char_vocab = _construct_char_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                           include_word_start_end=include_word_start_end)
        self.char_pad_index = self.char_vocab.padding_idx
        # Index vocab.
        self.max_word_len = max(len(word) for word in vocab.word_count.keys())
        if include_word_start_end:
            self.max_word_len += 2
        self.fill = P.Fill()
        self.zeros = P.Zeros()
        self.words_to_chars_embedding = np.full((len(vocab), self.max_word_len), self.char_pad_index, dtype=np.int32)
        self.word_lengths = np.zeros((len(vocab)), dtype=float)
        for word, index in tqdm(vocab.word2idx.items(), desc='Char_embedding processing:'):
            # if index!=vocab.padding_idx:
            # If it is a pad, it is directly pad_value. Modified to not distinguish pads, so that all <pad>s are also
            # the same embed.
            if include_word_start_end:
                word = ['<bow>'] + list(word) + ['<eow>']
            self.words_to_chars_embedding[index, :len(word)] = np.array([self.char_vocab[c] for c in word],
                                                                        dtype=np.int32)
        self.words_to_chars_embedding = mindspore.Tensor(self.words_to_chars_embedding, dtype=mstype.int32)

        if isinstance(pre_train_char_embed, str):
            self.char_embedding = StaticEmbedding(self.char_vocab, model_dir_or_name=pre_train_char_embed)
        else:
            self.char_embedding = nn.Embedding(vocab_size=len(self.char_vocab),
                                               embedding_size=self.char_emb_size, padding_idx=0)

        self.conv_net = nn.CellList([nn.Conv1d(
            self.char_embedding.embedding_size, filter_nums[i], kernel_size=kernel_sizes[i], has_bias=True,
            pad_mode='pad', padding=kernel_sizes[i] // 2) for i in range(len(kernel_sizes))])
        self._embed_size = embed_size
        self.requires_grad = requires_grad
        self.tile = P.Tile()
        self.select = P.Select()
        self.equal = P.Equal()
        self.concat = P.Concat(axis=1)
        self.cast = P.Cast()
        self.slice = P.Slice()
        self.expand_dim = P.ExpandDims()
        self.dropout = nn.Dropout(keep_prob=0.5)

        self.char_encoders = nn.CellList([nn.Conv3d(in_channels=1, out_channels=filter_nums[i],
                                                    kernel_size=(1, kernel_sizes[i], char_emb_size), has_bias=False
                                                    , stride=(1, 1, 30)) for i in range(len(kernel_sizes))])

    def construct(self, words: mindspore.Tensor) -> mindspore.Tensor:
        """
        Input the index of words, then output the char embedding representations of the corresponding words.

        Args:
            words (Tensor): The shape is (batch_size, max_len). The index of words.

        Return:
            chars_emb (Tensor): The shape is (batch_size, max_len, embed_size). The char embedding of the words.
        """
        _, max_len = words.shape
        chars_tokens = self.words_to_chars_embedding[words]  # (batch_size, max_len, max_word_len)
        chars_tokens = chars_tokens.reshape(-1, max_len * self.max_word_len)
        chars_emb = self.char_embedding(chars_tokens)
        chars_emb = self.dropout(chars_emb)
        chars_emb = chars_emb.reshape(-1, 1, max_len, self.max_word_len, self.char_emb_size)
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(chars_emb)
            pool_output = conv_output.max(-2)
            pool_output = pool_output.squeeze()
            char_conv_outputs.append(pool_output)

        char_conv_outputs = self.concat(char_conv_outputs)
        char_conv_outputs = char_conv_outputs.transpose(0, 2, 1)
        return char_conv_outputs
