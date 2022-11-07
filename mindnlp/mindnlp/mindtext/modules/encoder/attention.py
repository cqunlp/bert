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
"""Attention classes for Encoder."""

import math
from typing import Optional

import mindspore
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer as init
from mindspore.common.initializer import XavierUniform as xavUniform
from mindspore import numpy as msnp

__all__ = [
    "DotAttention",
    "MultiHeadAttention",
    "SelfAttention",
]


class CastWrapper(nn.Cell):
    """
    Cast wrapper.
    """

    def __init__(self, src_type: mindspore.dtype = mstype.float32, dst_type: mindspore.dtype = mstype.float32):
        super(CastWrapper, self).__init__()
        self.cast = P.Cast()
        self.scr_type = src_type
        self.dst_type = dst_type

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Args:
            x (Tensor): The shape of tensor is (x1,x2,...,xR). The tensor to be cast.

        Returns:
            Tensor, the shape of tensor is the same as x.
        """
        return self.cast(x, self.dst_type)


class LayerPreprocess(nn.Cell):
    """
    Preprocess input of each layer.

    Args:
        in_channels (int): The size of input channel, generally, last dim of input.

    """

    def __init__(self, in_channels: Optional[int] = None):
        super(LayerPreprocess, self).__init__()
        self.layernorm = nn.LayerNorm((in_channels,))
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def construct(self, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Args:
            input_tensor (Tensor): The input of preprocess layer.

        Returns:
            outputs (Tensor): The output of preprocess layer.
        """
        output = self.cast(input_tensor, mstype.float32)
        output = self.layernorm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output


class LayerPostprocess(nn.Cell):
    """
    Postprocess output of each layer.

    Args:
        dropout_prob (float): The dropout probability for postprocess layer. Default: 0.1.

    """

    def __init__(self,
                 dropout_prob: float = 0.1):
        super(LayerPostprocess, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Args:
            hidden_tensor (Tensor): The output of hidden layer.
            input_tensor (Tensor): The input of hidden layer.

        Returns:
            output (Tensor): The output of postprocess layer.
        """
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class DotAttention(nn.Cell):
    """
    DotAttention in Transformer.

    Args:
        key_size (int): The size of last dim of Key.
        value_size (int): The size of last dim of Value.
        dropout (int): The dropout rate of outputs. Default: 0.0.
    """

    def __init__(self, key_size: int, value_size: int, dropout: float = 0.0, has_attn_mask: bool = False):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(keep_prob=1 - dropout)
        self.has_attn_mask = has_attn_mask
        self.softmax = nn.Softmax(axis=-1)
        self.matmul = nn.MatMul()
        self.select = P.Select()

    def construct(self, q: mindspore.Tensor, k: mindspore.Tensor, v: mindspore.Tensor,
                  attn_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Args:
            q (Tensor): The shape is (batch_size, q_len, q_size). The queries for Attention.
            k (Tensor):  The shape is (batch_size, k_len, k_size). The keys for Attention.
            v (Tensor): The shape is (batch_size, v_len, v_size). The values for Attention.
            attn_mask (Tensor): The is shape (batch_size, q_len, q_len). The mask matrix for Attention,
                                the values should be True or False. Default: None.

        Returns:
            output (Tensor): The output of DotAttention.
        """
        attn = self.matmul(q, k.transpose((0, 2, 1))) / self.scale
        if self.has_attn_mask:
            attn_mask = attn_mask.astype(mstype.bool_)
            mask_full = msnp.full_like(attn, -1e9)
            attn = self.select(attn_mask, mask_full, attn)
        attn = self.softmax(attn)
        attn = self.drop(attn)
        output = self.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        batch_size (int): Batch size of input datasets.
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        out_act (str): Activation function for the output transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      MultiHeadAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        compute_type (class:`mindspore.dtype`): Compute type in MultiHeadAttention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 from_tensor_width: int,
                 to_tensor_width: int,
                 out_tensor_width: int,
                 num_attention_heads: int = 1,
                 size_per_head: int = 512,
                 query_act: Optional[str] = None,
                 key_act: Optional[str] = None,
                 value_act: Optional[str] = None,
                 out_act: Optional[str] = None,
                 has_attention_mask: bool = True,
                 attention_probs_dropout_prob: float = 0.0,
                 use_one_hot_embeddings: bool = False,
                 initializer_range: float = 0.02,
                 do_return_2d_tensor: bool = True,
                 compute_type: mindspore.dtype = mstype.float32):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = batch_size
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor

        self.scores_mul = mindspore.Tensor([1.0 / math.sqrt(float(self.size_per_head))], dtype=compute_type)
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    has_bias=False,
                                    weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  has_bias=False,
                                  weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    has_bias=False,
                                    weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)
        self.out_layer = nn.Dense(units,
                                  out_tensor_width,
                                  activation=out_act,
                                  has_bias=False,
                                  weight_init=init(xavUniform(), [units, to_tensor_width])).to_float(compute_type)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = mindspore.Tensor([-10000.0,], dtype=compute_type)
        self.batch_num = batch_size * num_attention_heads
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        self.cast_compute_type = CastWrapper(dst_type=compute_type)
        self.softmax_cast = P.Cast()

    def construct(self, from_tensor: mindspore.Tensor, to_tensor: mindspore.Tensor,
                  seq_length: int, enc_seq_length: int,
                  attention_mask: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Apply multi-head attention.

        Args:
            from_tensor (Tensor): The shape is (batch_size, from_seq_len, dim). The from_tensor sequence, generally
                                it's query tensor(Q) for attention.
            to_tensor (Tensor): The shape is (batch_size, to_seq_len, dim). The to_tensor sequences, generally it's key
                                tensor(K) and value tensor(V) for attention, K = V.
            seq_length (int): The length of from_tensor.
            enc_seq_length (int): The length of to_tensor.
            attention_mask (Tensor): The shape is (from_seq_len, to_seq_len) or (batch_size, from_seq_len, to_seq_len).
                                The mask matrix(2D or 3D) for attention, the values should be [0/1] or [True/False].
                                Default: None.

        Returns:
            output (Tensor): The output of multi-head attention.
        """
        from_seq_length = seq_length
        to_seq_length = enc_seq_length
        shape_from = (self.batch_size, from_seq_length, self.num_attention_heads, self.size_per_head)
        shape_to = (self.batch_size, to_seq_length, self.num_attention_heads, self.size_per_head)
        if self.do_return_2d_tensor:
            shape_return = (self.batch_size * from_seq_length, self.num_attention_heads * self.size_per_head)
            if from_seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (self.batch_size, from_seq_length, self.num_attention_heads * self.size_per_head)

        # reshape 2d/3d input tensors to 2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float32)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key_layer))
        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, shape_return)
        context_layer = self.out_layer(context_layer)
        return context_layer


class SelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        batch_size (int): Batch size of input dataset.
        hidden_size (int): Size of attention layers.
        num_attention_heads (int): Number of attention heads. Default: 16.
        attention_probs_dropout_prob (float): The dropout probability for
                                      SelfAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        has_attention_mask (bool): Specifies whether has attention mask. Default: True.
        is_encdec_att (bool): Specifies whether query sequence and memory sequence are different. Default: False.
        compute_type (class:`mindspore.dtype`): Compute type in MultiHeadAttention. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size: int,
                 hidden_size: int,
                 num_attention_heads: int = 16,
                 attention_probs_dropout_prob: float = 0.1,
                 use_one_hot_embeddings: bool = False,
                 initializer_range: float = 0.02,
                 hidden_dropout_prob: float = 0.1,
                 has_attention_mask: bool = True,
                 is_encdec_att: bool = False,
                 compute_type: mindspore.dtype = mstype.float32):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.size_per_head = int(hidden_size / num_attention_heads)
        self.is_encdec_att = is_encdec_att

        self.attention = MultiHeadAttention(
            batch_size=batch_size,
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            out_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.preprocess = LayerPreprocess(in_channels=hidden_size)
        self.postprocess = LayerPostprocess(dropout_prob=hidden_dropout_prob)

        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor: mindspore.Tensor, memory_tensor: mindspore.Tensor,
                  attention_mask: mindspore.Tensor, seq_length: int,
                  enc_seq_length: int) -> mindspore.Tensor:
        """
        Args:
            input_tensor (Tensor): The shape is (batch_size, seq_len, hidden_units). The input_tensor sequence,
                                    generally it's query tensor(Q) for self-attention.
            memory_tensor (Tensor): The shape is (batch_size, seq_len, hidden_units). The memory_tensor sequence,
                                    generally it's key tensor(K) and value tensor(V) for self-attention, K = V.
            attention_mask (Tensor): The shape is (from_seq_len, to_seq_len) or (batch_size, from_seq_len, to_seq_len).
                                The mask matrix(2D or 3D) for attention, the values should be [0/1] or [True/False].
                                Default: None.
            seq_length (int): The length of input_tensor.
            enc_seq_length (int): The length of memory_tensor.

        Returns:
            output (Tensor): The output of self-attention.
        """
        input_tensor = self.reshape(input_tensor, self.shape)
        memory_tensor = self.reshape(memory_tensor, self.shape)

        output = self.preprocess(input_tensor)

        if not self.is_encdec_att:
            memory_tensor = output

        attention_output = self.attention(output, memory_tensor, seq_length, enc_seq_length, attention_mask)
        output = self.postprocess(attention_output, input_tensor)
        return output
