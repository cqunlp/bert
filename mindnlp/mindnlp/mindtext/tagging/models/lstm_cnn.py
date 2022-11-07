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
"""LSTM-CNNs Model for sequence tagging."""
from typing import Union, List, Optional

import mindspore
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype
from mindspore import ParameterTuple
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import GradOperation
from mindspore.communication import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindtext.common.data import Vocabulary
from mindtext.embeddings.static_embedding import StaticEmbedding
from mindtext.embeddings.char_embedding import CNNCharEmbedding
from mindtext.modules.decoder.norm_decoder import NormalDecoder


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


class LstmCnnConfig(nn.Cell):
    """
    Config of LSTM-CNN model.

    Args:
        vocab (Vocabulary): The vocabulary for word embedding and char embedding.
        model_dir_or_name (Union[str, None]): The model file path for pretrained embedding, such as Google Glove.
        word_embed_size (int): The embedding size of word embedding. Default: 100.
        char_embed_size (int): The embedding size of char embedding. Default: 100.
        embed_dropout (float): The dropout rate of the embedding. Default: 0.1.
        filter_nums (List[int]): The number of filters. The length needs to be consistent with the kernels.
                                Default: [40, 30, 20].
        conv_kernel_sizes (List[int]): The size of kernel. Default: [5, 3, 1].
        pool_method (str): The pool method used when synthesizing the representation of the character into a
                            representation, supports 'avg' and 'max'. Default: max.
        conv_char_embed_activation (str): The activation method used after CNN, supports 'relu','sigmoid','tanh' or
                                            custom functions.
        min_char_freq (int): The minimum frequency of occurrence of a character. Default: 2.
        num_layers (int): The number of layers of bi_lstm cell. Default: 2.
        hidden_size (int): The size of hidden layers of bi_lstm cell. Default: 100.
        output_size (int): The size of output of LSTM-CNNs model. Default: 100.
        num_classes (int): The number of class. Default: 9.(When dataset is conll2003.)
        hidden_activation (str): The activation function of hidden layer (linear decoder). Default: 'relu'.
        hidden_dropout (float): The dropout rate of bi_lstm and linear decoder. Default: 0.1.
        embed_requires_grad (bool): Whether to update the weight.
        pre_train_char_embed (str): There are two ways to call the pre-trained character embedding: the first is to pass
                                    in the embedding folder (there should be only one file with .txt as the suffix) or
                                    the file path. The second is to pass in the name of the embedding. In the second
                                    case, it will automatically check whether the model exists in the cache, if not,
                                    it will be downloaded automatically. If the input is None, use the dimension of
                                    embedding_dim to randomly initialize an embedding.
        include_word_start_end (bool): Whether to add special marking symbols before and ending the character at the
                                        beginning and end of each word.

    """

    def __init__(self,
                 vocab: Vocabulary,
                 model_dir_or_name: Union[str, None] = None,
                 word_embed_size: int = 100,
                 char_embed_size: int = 30,
                 embed_dropout: float = 0.69,
                 filter_nums: List[int] = (30,),
                 conv_kernel_sizes: List[int] = (3,),
                 pool_method: str = 'avg',
                 conv_char_embed_activation: str = 'relu',
                 min_char_freq: int = 1,
                 num_layers: int = 2,
                 hidden_size: int = 250,
                 output_size: int = 100,
                 num_classes: int = 9,
                 hidden_activation: str = 'relu',
                 hidden_dropout: float = 0.68,
                 embed_requires_grad: bool = True,
                 pre_train_char_embed: Optional[str] = None,
                 include_word_start_end: bool = True
                 ):
        super(LstmCnnConfig, self).__init__()
        self.vocab = vocab
        self.model_dir_or_name = model_dir_or_name
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.embed_dropout = embed_dropout
        self.embed_requires_grad = embed_requires_grad
        self.filter_nums = filter_nums
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_method = pool_method
        self.conv_char_embed_activation = conv_char_embed_activation
        self.min_char_freq = min_char_freq
        self.pre_train_char_embed = pre_train_char_embed
        self.include_word_start_end = include_word_start_end
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = word_embed_size + char_embed_size
        self.hidden_dropout = hidden_dropout
        self.output_size = output_size
        self.num_classes = num_classes
        self.hidden_activation = hidden_activation


class LstmCnn(nn.Cell):
    """
    LSTM-CNNs model.

    Args:
        config (LstmCnnConfig): The config object for LSTM-CNNs model.
    """

    def __init__(self, config: LstmCnnConfig):
        super(LstmCnn, self).__init__()
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        # Word embedding.
        self.word_embedding = StaticEmbedding(vocab=config.vocab, model_dir_or_name=None,
                                              embedding_dim=config.word_embed_size,
                                              requires_grad=config.embed_requires_grad,
                                              dropout=config.embed_dropout)

        # # CNN char embedding.
        self.cnn_char_embedding = CNNCharEmbedding(vocab=config.vocab,
                                                   char_emb_size=config.char_embed_size,
                                                   dropout=config.embed_dropout, filter_nums=config.filter_nums,
                                                   kernel_sizes=config.conv_kernel_sizes,
                                                   pool_method=config.pool_method,
                                                   activation=config.conv_char_embed_activation)

        # bi_lstm
        self.bi_lstm = nn.LSTM(input_size=130, hidden_size=config.hidden_size,
                               num_layers=config.num_layers, batch_first=True, bidirectional=True)

        # Output layer (Decoder): linear + log_softmax
        # NormalDecoder is a linear decoder.
        self.liner_decoder = NormalDecoder(num_filters=config.hidden_size * 2, num_classes=config.num_classes,
                                           classes_dropout=config.hidden_dropout, activation=None)

        # log_softmax
        self.log_softmax = nn.LogSoftmax()

        # Utils for LSTM-CNNs.
        self.dropout_emb = nn.Dropout(keep_prob=1 - config.embed_dropout)
        self.dropout = nn.Dropout(keep_prob=1 - config.hidden_dropout)
        self.concat = P.Concat(axis=-1)
        self.zeros = P.Zeros()
        self.tanh = P.Tanh()

    def construct(self, words: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply LSTM-CNNs model.

        Args:
            words (Tensor): The shape is (batch_size, max_len). The index of words.

        Returns:
            output (Tensor): The shape is (batch_size, max_len, output_size). The output of LSTM-CNNs model.
        """
        batch_size = words.shape[0]
        word_emb = self.word_embedding(words)
        char_emb = self.cnn_char_embedding(words)
        char_emb = self.dropout_emb(char_emb)
        word_char_emb = self.concat((word_emb, char_emb))
        word_char_emb = self.dropout_emb(word_char_emb)

        h0 = self.zeros((2 * self.num_layers, batch_size, self.hidden_size), mstype.float32)
        c0 = self.zeros((2 * self.num_layers, batch_size, self.hidden_size), mstype.float32)
        output, _ = self.bi_lstm(word_char_emb, (h0, c0))
        output = self.dropout(output)
        output = self.tanh(output)
        output = self.liner_decoder(output)

        return output


class LstmCnnWithLoss(nn.Cell):
    """
    Provide LstmCnn training loss

    Args:
        net(nn.Cell): LstmCnn model.
        loss(Loss): LstmCnn loss.
    """

    def __init__(self, net, loss):
        super(LstmCnnWithLoss, self).__init__()
        self.lstmcnn = net
        self.loss_func = loss
        self.squeeze = P.Squeeze(axis=1)
        self.argmax = P.ArgMaxWithValue(axis=-1)

    def construct(self, src_tokens, src_mask, label_idx):
        """
        Apply LSTM-CNNs model.

        Args:
            src_tokens (Tensor): Tokens of sentences
            src_mask (Tensor): Masks of sentences
            label_idx: Labels of sentences, such as NER labels

        Returns:
            loss : The shape is (batch_size, max_len, output_size). The output of LSTM-CNNs model.
        """
        predict_score = self.lstmcnn(src_tokens)
        if self.training:
            loss = self.loss_func(predict_score.reshape(-1, predict_score.shape[-1]), label_idx.reshape(-1))
            loss = loss.reshape(src_tokens.shape[0], -1)
            loss *= src_mask
            loss = loss.sum(axis=-1)
            loss = loss.mean()
        else:
            loss = predict_score
        return loss


class LstmCnnTrainOneStep(nn.Cell):
    """
    LstmCnn train class.

    Args:
        net(nn.Cell):
        loss(Loss): LstmCnn loss.
        optimizer(Optimizer): LstmCnn optimizer.
    """

    def __init__(self, net, optimizer, sens=1.0):
        super(LstmCnnTrainOneStep, self).__init__(auto_prefix=False)
        self.network = net
        self.network.init_parameters_data()
        self.optimizer = optimizer
        self.weights = ParameterTuple(self.network.trainable_params())
        self.grad = GradOperation(get_by_list=True)
        self.sens = sens
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode not in ParallelMode.MODE_LIST:
            raise ValueError("Parallel mode does not support: ", self.parallel_mode)
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = C.HyperMap()
        self.cast = P.Cast()

    def construct(self, src_token_text, src_mask, label_idx_tag):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(src_token_text, src_mask, label_idx_tag)
        grads = self.grad(self.network, weights)(src_token_text,
                                                 src_mask,
                                                 label_idx_tag)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)
