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
    Encoder classes for LUKE.
"""
import math

import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
import mindspore.ops.operations as P
import mindspore.ops.composite as C
import mindspore.numpy as np

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, TruncatedNormal

from mindtext.embeddings.luke_embeddings import RobertaEmbeddings, EntityEmbeddings

grad_scale = mindspore.ops.MultitypeFuncGraph("grad_scale")


class LukeConfig:
    """
    Configuration for `BertModel`.

    Args:
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 32000.
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        num_hidden_layers (int): Number of hidden layers in the BertTransformer encoder
                           cell. Default: 12.
        num_attention_heads (int): Number of attention heads in the BertTransformer
                             encoder cell. Default: 12.
        intermediate_size (int): Size of intermediate layer in the BertTransformer
                           encoder cell. Default: 3072.
        hidden_act (str): Activation function used in the BertTransformer encoder
                    cell. Default: "gelu".
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        type_vocab_size (int): Size of token type vocab. Default: 16.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """

    def __init__(self,
                 seq_length=256,
                 vocab_size=50265,
                 hidden_size=1024,
                 num_hidden_layers=24,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=514,
                 type_vocab_size=1,
                 initializer_range=0.02,
                 batch_size=2,
                 use_relative_positions=False,
                 dtype=mstype.float32,
                 compute_type=mstype.float32,
                 bert_model_name="roberta-large",
                 bos_token_id=0,
                 entity_emb_size=256,
                 entity_vocab_size=500000,
                 eos_token_id=2,
                 gradient_checkpointing=False,
                 layer_norm_eps=1e-05,
                 model_type="luke",
                 output_past=True,
                 pad_token_id=1,
                 position_embedding_type="absolute",
                 ):
        self.bos_token_id = bos_token_id
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.use_relative_positions = use_relative_positions
        self.dtype = dtype
        self.compute_type = compute_type
        self.bert_model_name = bert_model_name
        self.entity_emb_size = entity_emb_size
        self.entity_vocab_size = entity_vocab_size
        self.eos_token_id = eos_token_id
        self.gradient_checkpointing = gradient_checkpointing
        self.layer_norm_eps = layer_norm_eps
        self.model_type = model_type
        self.output_past = output_past
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.embedding_size = self.hidden_size


class LukeModel(nn.Cell):
    """LukeModel"""

    def __init__(self, config):
        super(LukeModel, self).__init__()
        self.encoder = BertEncoderCell(config.batch_size)
        self.pooler = nn.Dense(config.hidden_size, config.hidden_size,
                               activation="tanh",
                               weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_type)
        self.embeddings = RobertaEmbeddings(config)
        self.embeddings.token_type_embeddings.requires_grad = False
        self.entity_embeddings = EntityEmbeddings(config)


def _compute_extended_attention_mask(word_attention_mask, entity_attention_mask):
    """_compute_extended_attention_mask"""
    attention_mask = word_attention_mask

    op_concat = ops.Concat(1)
    attention_mask = op_concat((attention_mask, entity_attention_mask))

    unsqueezee = ops.ExpandDims()
    extended_attention_mask = unsqueezee(unsqueezee(attention_mask, 1), 2)
    extended_attention_mask = extended_attention_mask.astype(mstype.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class LukeEntityAwareAttentionModel(nn.Cell):
    """LukeEntityAwareAttentionModel"""

    def __init__(self, config):
        super(LukeEntityAwareAttentionModel, self).__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.entity_embeddings = EntityEmbeddings(config)
        self.encoder = EntityAwareEncoder(config)

    def construct(self, word_ids, word_segment_ids, word_attention_mask, entity_ids, entity_position_ids,
                  entity_segment_ids, entity_attention_mask):
        word_embeddings = self.embeddings(word_ids, word_segment_ids)
        entity_embeddings = self.entity_embeddings(entity_ids, entity_position_ids, entity_segment_ids)
        attention_mask = _compute_extended_attention_mask(word_attention_mask, entity_attention_mask)
        output = self.encoder(word_embeddings, entity_embeddings, attention_mask)

        return output


class EntityAwareSelfAttention(nn.Cell):
    """EntityAwareSelfAttention"""

    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Dense(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Dense(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(1 - config.attention_probs_dropout_prob)
        self.concat = ops.Concat(1)
        self.concat2 = ops.Concat(2)
        self.concat3 = ops.Concat(3)
        self.sotfmax = ops.Softmax()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = ops.shape(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        out = self.reshape(x, new_x_shape)
        out = self.transpose(out, (0, 2, 1, 3))
        return out

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """EntityAwareSelfAttention construct"""
        word_size = self.shape(word_hidden_states)[1]
        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))
        key_layer = self.transpose_for_scores(self.key(self.concat([word_hidden_states, entity_hidden_states])))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = ops.matmul(w2w_query_layer, ops.transpose(w2w_key_layer, (0, 1, 3, 2)))
        w2e_attention_scores = ops.matmul(w2e_query_layer, ops.transpose(w2e_key_layer, (0, 1, 3, 2)))
        e2w_attention_scores = ops.matmul(e2w_query_layer, ops.transpose(e2w_key_layer, (0, 1, 3, 2)))
        e2e_attention_scores = ops.matmul(e2e_query_layer, ops.transpose(e2e_key_layer, (0, 1, 3, 2)))

        word_attention_scores = self.concat3([w2w_attention_scores, w2e_attention_scores])
        entity_attention_scores = self.concat3([e2w_attention_scores, e2e_attention_scores])
        attention_scores = self.concat2([word_attention_scores, entity_attention_scores])

        attention_scores = attention_scores / np.sqrt(1.0 * self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(self.concat([word_hidden_states, entity_hidden_states]))
        )
        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = ops.transpose(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = ops.shape(context_layer)[:-2] + (self.all_head_size,)
        context_layer = self.reshape(context_layer, new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class EntityAwareAttention(nn.Cell):
    """EntityAwareAttention"""

    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self_attention = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.concat = ops.Concat(1)

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """
        EntityAwareAttention's construct
        """
        word_self_output, entity_self_output = self.self_attention.construct(word_hidden_states, entity_hidden_states,
                                                                             attention_mask)

        hidden_states = self.concat([word_hidden_states, entity_hidden_states])
        self_output = self.concat([word_self_output, entity_self_output])
        out = self.output.construct(self_output, hidden_states)

        out1 = out[:, : ops.shape(word_hidden_states)[1], :]
        out2 = out[:, ops.shape(word_hidden_states)[1]:, :]
        return out1, out2


class EntityAwareLayer(nn.Cell):
    """EntityAwareLayer"""

    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = nn.Dense(config.hidden_size,
                                     config.intermediate_size,
                                     activation=config.hidden_act,
                                     weight_init=TruncatedNormal(config.initializer_range)).to_float(mstype.float32)
        self.output = BertOutput(config.intermediate_size, config.hidden_size)
        self.concat = ops.Concat(1)

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """
        EntityAwareLayer's construct
        """
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )

        attention_output = self.concat([word_attention_output, entity_attention_output])
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        l_out1 = layer_output[:, : ops.shape(word_hidden_states)[1], :]
        l_out2 = layer_output[:, ops.shape(word_hidden_states)[1]:, :]
        return l_out1, l_out2


class EntityAwareEncoder(nn.Cell):
    """EntityAwareEncoder"""

    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.CellList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(self, word_hidden_states, entity_hidden_states, attention_mask):
        """
            EntityAwareEncoder's construct
        """
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )

        return word_hidden_states, entity_hidden_states


class BertOutput(nn.Cell):
    """
    Apply a linear computation to hidden status and a residual computation to input.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        dropout_prob (float): The dropout probability. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(BertOutput, self).__init__()
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.dropout_prob = dropout_prob
        self.add = P.Add()
        self.layernorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        self.cast = P.Cast()

    def construct(self, hidden_status, input_tensor):
        output = self.dense(hidden_status)
        output = self.dropout(output)
        output = self.add(input_tensor, output)
        output = self.layernorm(output)
        return output


class BertEncoderCell(nn.Cell):
    """
    Encoder cells used in BertTransformer.

    Args:
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 512.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.02.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        hidden_act (str): Activation function. Default: "gelu".
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """

    def __init__(self,
                 hidden_size=1024,
                 seq_length=512,
                 num_attention_heads=16,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 compute_type=mstype.float32):
        super(BertEncoderCell, self).__init__()
        self.attention = BertSelfAttention(
            hidden_size=hidden_size,
            seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            use_relative_positions=use_relative_positions,
            compute_type=compute_type)
        self.intermediate = nn.Dense(in_channels=hidden_size,
                                     out_channels=intermediate_size,
                                     activation=hidden_act,
                                     weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.output = BertOutput(in_channels=intermediate_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type)

    def construct(self, hidden_states, attention_mask):
        # self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        # feed construct
        intermediate_output = self.intermediate(attention_output)
        # add and normalize
        output = self.output(intermediate_output, attention_output)
        return output


class BertSelfOutput(nn.Cell):
    """
    BertSelfOutput class
    """
    def __init__(self, config, compute_type=mstype.float32):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size,
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(compute_type)
        self.layernorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps).to_float(compute_type)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.add = P.Add()

    def construct(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.add(input_tensor, hidden_states)
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Cell):
    """
    Apply self-attention.

    Args:
        seq_length (int): Length of input sequence.
        hidden_size (int): Size of the bert encoder layers.
        num_attention_heads (int): Number of attention heads. Default: 12.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        use_one_hot_embeddings (bool): Specifies whether to use one_hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertSelfAttention. Default: mstype.float32.
    """

    def __init__(self,
                 seq_length,
                 hidden_size=768,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 compute_type=mstype.float32):
        super(BertSelfAttention, self).__init__()
        hidden_size = 1024
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        self.size_per_head = int(hidden_size / num_attention_heads)

        self.attention = BertAttention(
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            from_seq_length=seq_length,
            to_seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            use_relative_positions=use_relative_positions,
            has_attention_mask=True,
            do_return_2d_tensor=True,
            compute_type=compute_type)

        self.output = BertOutput(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type)
        self.reshape = P.Reshape()
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask):
        input_tensor = self.reshape(input_tensor, self.shape)
        attention_output = self.attention(input_tensor, input_tensor, attention_mask)
        output = self.output(attention_output, input_tensor)
        return output


class BertAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        from_seq_length (int): Length of from_tensor sequence.
        to_seq_length (int): Length of to_tensor sequence.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor (bool): True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertAttention. Default: mstype.float32.
    """

    def __init__(self,
                 from_tensor_width,
                 to_tensor_width,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 has_attention_mask=False,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 use_relative_positions=False,
                 compute_type=mstype.float32):

        super(BertAttention, self).__init__()
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        self.use_relative_positions = use_relative_positions

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        self.reshape = P.Reshape()
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        weight = TruncatedNormal(initializer_range)
        units = num_attention_heads * size_per_head
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    weight_init=weight).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  weight_init=weight).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    weight_init=weight).to_float(compute_type)

        self.shape_from = (-1, from_seq_length, num_attention_heads, size_per_head)
        self.shape_to = (-1, to_seq_length, num_attention_heads, size_per_head)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        self.multiply_data = -10000.0
        self.matmul = P.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1 - attention_probs_dropout_prob)

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()
        if do_return_2d_tensor:
            self.shape_return = (-1, num_attention_heads * size_per_head)
        else:
            self.shape_return = (-1, from_seq_length, num_attention_heads * size_per_head)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        if self.use_relative_positions:
            self._generate_relative_positions_embeddings = \
                RelaPosEmbeddingsGenerator(length=to_seq_length,
                                           depth=size_per_head,
                                           max_relative_position=16,
                                           initializer_range=initializer_range,
                                           use_one_hot_embeddings=use_one_hot_embeddings)

    def construct(self, from_tensor, to_tensor, attention_mask):
        """reshape 2d/3d input tensors to 2d"""
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, self.shape_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, self.shape_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # relations_keys is [F|T, F|T, H]
            relations_keys = self._generate_relative_positions_embeddings()
            relations_keys = self.cast_compute_type(relations_keys)
            # query_layer_t is [F, B, N, H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r is [F, B * N, H]
            query_layer_r = self.reshape(query_layer_t,
                                         (self.from_seq_length,
                                          -1,
                                          self.size_per_head))
            # key_position_scores is [F, B * N, F|T]
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r is [F, B, N, F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (self.from_seq_length,
                                                  -1,
                                                  self.num_attention_heads,
                                                  self.from_seq_length))
            # key_position_scores_r_t is [B, N, F, F|T]
            key_position_scores_r_t = self.transpose(key_position_scores_r,
                                                     self.trans_shape_position)
            attention_scores = attention_scores + key_position_scores_r_t

        attention_scores = self.multiply(self.scores_mul, attention_scores)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.reshape(value_out, self.shape_to)
        value_layer = self.transpose(value_layer, self.trans_shape)
        context_layer = self.matmul(attention_probs, value_layer)

        # use_relative_position, supplementary logic
        if self.use_relative_positions:
            # relations_values is [F|T, F|T, H]
            relations_values = self._generate_relative_positions_embeddings()
            relations_values = self.cast_compute_type(relations_values)
            # attention_probs_t is [F, B, N, T]
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            # attention_probs_r is [F, B * N, T]
            attention_probs_r = self.reshape(
                attention_probs_t,
                (self.from_seq_length,
                 -1,
                 self.to_seq_length))
            # value_position_scores is [F, B * N, H]
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # value_position_scores_r is [F, B, N, H]
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (self.from_seq_length,
                                                    -1,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            # value_position_scores_r_t is [B, N, F, H]
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            context_layer = context_layer + value_position_scores_r_t

        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        return context_layer


class SaturateCast(nn.Cell):
    """
    Performs a safe saturating cast. This operation applies proper clamping before casting to prevent
    the danger that the value will overflow or underflow.

    Args:
        src_type (:class:`mindspore.dtype`): The type of the elements of the input tensor. Default: mstype.float32.
        dst_type (:class:`mindspore.dtype`): The type of the elements of the output tensor. Default: mstype.float32.
    """

    def __init__(self, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        np_type = mstype.dtype_to_nptype(dst_type)

        self.tensor_min_type = float(np.finfo(np_type).min)
        self.tensor_max_type = float(np.finfo(np_type).max)

        self.min_op = P.Minimum()
        self.max_op = P.Maximum()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        out = self.max_op(x, self.tensor_min_type)
        out = self.min_op(out, self.tensor_max_type)
        return self.cast(out, self.dst_type)


class RelaPosEmbeddingsGenerator(nn.Cell):
    """
    Generates tensor of size [length, length, depth].

    Args:
        length (int): Length of one dim for the matrix to be generated.
        depth (int): Size of each attention head.
        max_relative_position (int): Maxmum value of relative position.
        initializer_range (float): Initialization value of TruncatedNormal.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """

    def __init__(self,
                 length,
                 depth,
                 max_relative_position,
                 initializer_range,
                 use_one_hot_embeddings=False):
        super(RelaPosEmbeddingsGenerator, self).__init__()
        self.depth = depth
        self.vocab_size = max_relative_position * 2 + 1
        self.use_one_hot_embeddings = use_one_hot_embeddings

        self.embeddings_table = Parameter(
            initializer(TruncatedNormal(initializer_range),
                        [self.vocab_size, self.depth]))

        self.relative_positions_matrix = RelaPosMatrixGenerator(length=length,
                                                                max_relative_position=max_relative_position)
        self.reshape = P.Reshape()
        self.one_hot = nn.OneHot(depth=self.vocab_size)
        self.shape = P.Shape()
        self.gather = P.Gather()  # index_select
        self.matmul = P.BatchMatMul()

    def construct(self):
        """Generate embedding for each relative position of dimension depth."""
        relative_positions_matrix_out = self.relative_positions_matrix()

        if self.use_one_hot_embeddings:
            flat_relative_positions_matrix = self.reshape(relative_positions_matrix_out, (-1,))
            one_hot_relative_positions_matrix = self.one_hot(
                flat_relative_positions_matrix)
            embeddings = self.matmul(one_hot_relative_positions_matrix, self.embeddings_table)
            my_shape = self.shape(relative_positions_matrix_out) + (self.depth,)
            embeddings = self.reshape(embeddings, my_shape)
        else:
            embeddings = self.gather(self.embeddings_table,
                                     relative_positions_matrix_out, 0)
        return embeddings


class RelaPosMatrixGenerator(nn.Cell):
    """
    Generates matrix of relative positions between inputs.

    Args:
        length (int): Length of one dim for the matrix to be generated.
        max_relative_position (int): Max value of relative position.
    """

    def __init__(self, length, max_relative_position):
        super(RelaPosMatrixGenerator, self).__init__()
        self._length = length
        self._max_relative_position = max_relative_position
        self._min_relative_position = -max_relative_position
        self.range_length = -length + 1

        self.tile = P.Tile()
        self.range_mat = P.Reshape()
        self.sub = P.Sub()
        self.expanddims = P.ExpandDims()
        self.cast = P.Cast()

    def construct(self):
        """Generates matrix of relative positions between inputs."""
        range_vec_row_out = self.cast(F.tuple_to_array(F.make_range(self._length)), mstype.int32)
        range_vec_col_out = self.range_mat(range_vec_row_out, (self._length, -1))
        tile_row_out = self.tile(range_vec_row_out, (self._length,))
        tile_col_out = self.tile(range_vec_col_out, (1, self._length))
        range_mat_out = self.range_mat(tile_row_out, (self._length, self._length))
        transpose_out = self.range_mat(tile_col_out, (self._length, self._length))
        distance_mat = self.sub(range_mat_out, transpose_out)

        distance_mat_clipped = C.clip_by_value(distance_mat,
                                               self._min_relative_position,
                                               self._max_relative_position)

        # Shift values to be >=0. Each integer still uniquely identifies a
        # relative position difference.
        final_mat = distance_mat_clipped + self._max_relative_position
        return final_mat
