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
"""Seq2seq encoder class."""

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer as init
from mindspore.common.initializer import XavierUniform as xavUniform
from .attention import LayerPreprocess, LayerPostprocess, SelfAttention

__all__ = [
    "TransformerEncoder",
]


class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward.

    Args:
        in_channels (int): Size of the input layer.
        hidden_size (int): Size of the hidden layer.
        out_channels (int): Size of the output layer.
        hidden_act (str): Name of the activation function. Default: relu.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in FeedForward. Default: mstype.float32.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_size: int,
                 out_channels: int,
                 hidden_act: str = "relu",
                 hidden_dropout_prob: float = 0.1,
                 compute_type: mindspore.dtype = mstype.float32):
        super(FeedForward, self).__init__()

        self.conv1 = nn.Dense(in_channels,
                              hidden_size,
                              activation=hidden_act,
                              weight_init=init(xavUniform(), [hidden_size, in_channels])).to_float(compute_type)
        self.conv2 = nn.Dense(hidden_size,
                              out_channels,
                              weight_init=init(xavUniform(), [out_channels, hidden_size])).to_float(compute_type)

        self.preprocess = LayerPreprocess(in_channels=in_channels)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = P.Reshape()
        self.shape = (-1, in_channels)
        self.dropout = nn.Dropout(1 - hidden_dropout_prob)
        self.use_dropout = hidden_dropout_prob > 0

    def construct(self, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Apply FeedForward.

        Args:
            input_tensor (Tensor): The input tensor for FeedForward Network.

        Returns:
            output (Tensor): The output tensor of FeedForward Network.
        """
        input_tensor = self.reshape(input_tensor, self.shape)
        output = self.preprocess(input_tensor)
        output = self.conv1(output)
        if self.use_dropout:
            output = self.dropout(output)
        output = self.conv2(output)
        output = self.postprocess(output, input_tensor)
        return output


class EncoderCell(nn.Cell):
    """
    Encoder cells used in Transformer.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers. Default: 1024.
        num_attention_heads (int): Number of attention heads. Default: 16.
        intermediate_size (int): Size of intermediate layer. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.1.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        hidden_act (str): Activation function. Default: "relu".
        compute_type (class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 hidden_size: int = 1024,
                 num_attention_heads: int = 16,
                 intermediate_size: int = 4096,
                 attention_probs_dropout_prob: float = 0.1,
                 use_one_hot_embeddings: bool = False,
                 initializer_range: float = 0.02,
                 hidden_dropout_prob: float = 0.1,
                 hidden_act: str = "relu",
                 compute_type: mindspore.dtype = mstype.float32):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            batch_size=batch_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            is_encdec_att=False,
            compute_type=compute_type)
        self.feedforward = FeedForward(
            in_channels=hidden_size,
            hidden_size=intermediate_size,
            out_channels=hidden_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, hidden_states: mindspore.Tensor, attention_mask: mindspore.Tensor,
                  seq_length: int) -> mindspore.Tensor:
        """
        Apply EncoderCell.

        Args:
            hidden_states (Tensor): The shape is (batch_size, seq_len, hidden_size). Hidden states of encoder layer.
            attention_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix
                                    (2D or 3D) for self-attention, the values should be [0/1] or [True/False].
            seq_length (int): The length of input sequence (`hidden_states`).

        return:
            output (Tensor): The output of EncoderCell.
        """
        # self-attention with ln, res
        attention_output = self.attention(hidden_states, hidden_states, attention_mask, seq_length, seq_length)
        # feed forward with ln, res
        output = self.feedforward(attention_output)
        return output


class TransformerEncoder(nn.Cell):
    """
    Multi-layer transformer encoder.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of the encoder layers.
        num_hidden_layers (int): Number of hidden layers in encoder cells.
        num_attention_heads (int): Number of attention heads in encoder cells. Default: 16.
        intermediate_size (int): Size of intermediate layer in encoder cells. Default: 4096.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1..
        hidden_act (str): Activation function used in the encoder cells. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int = 16,
                 intermediate_size: int = 4096,
                 attention_probs_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 hidden_dropout_prob: float = 0.1,
                 hidden_act: str = "relu",
                 compute_type: mindspore.dtype = mstype.float32):
        super(TransformerEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_hidden_layers):
            layer = EncoderCell(batch_size=batch_size,
                                hidden_size=hidden_size,
                                num_attention_heads=num_attention_heads,
                                intermediate_size=intermediate_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                initializer_range=initializer_range,
                                hidden_dropout_prob=hidden_dropout_prob,
                                hidden_act=hidden_act,
                                compute_type=compute_type)
            layers.append(layer)
        self.layers = nn.CellList(layers)

        self.layer_preprocess = LayerPreprocess(in_channels=hidden_size)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor: mindspore.Tensor, attention_mask: mindspore.Tensor,
                  seq_length: int) -> mindspore.Tensor:
        """
        Apply encoder.

        Args:
            input_tensor (Tensor): The shape is (batch_size, seq_len, hidden_size). The input sequence tensor of
                                Transformer Encoder.
            attention_mask (Tensor): The shape is (seq_len, seq_len) or (batch_size, seq_len, seq_len). The mask matrix
                                    (2D or 3D) for self-attention, the values should be [0/1] or [True/False].
            seq_length (int): The length of input sequence.

        return:
            output (Tensor): The output of Transformer Encoder.
        """
        out_shape = (self.batch_size, seq_length, self.hidden_size)
        prev_output = self.reshape(input_tensor, self.shape)

        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask, seq_length)
            prev_output = layer_output

        prev_output = self.layer_preprocess(prev_output)
        output = self.reshape(prev_output, out_shape)
        return output
