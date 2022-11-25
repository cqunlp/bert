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
    Encoder classes for BART.
"""
import os
import json
import math
from typing import Union, Optional, Dict
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from ..decoder.beam_search import BeamSearchDecoder


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = ops.composite.MultitypeFuncGraph("clip_grad")


class BartConfig:
    """
    Args:
        vocab_size (int):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            inputs_ids passed when calling BartModel, defaults to 50264.
        d_model (int):
            Dimensionality of the layers and the pooler layer of the BART model, defaults to 1024.
        encoder_layers (int):
            Number of encoder layers of the BART model. Defines the unmber of the encoder layers, defaults to 12.
        decoder_layers (int):
            Number of decoder layers of the BART model. Defines the unmber of the decoder layers, defaults to 12.
        encoder_attention_heads (int):
            Number of attention heads for each attention layer in the Transformer encoder of the BART model,
            defaults to 16.
        decoder_attention_heads (int):
            Number of attention heads for each attention layer in the Transformer decoder of the BART models,
            defaults to 16.
        decoder_ffn_dim (int):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder of the BART models,
            defaults to 4096.
        encoder_ffn_dim (int):
            Dimensionality of the "intermediate" (often named feed-forward)layer in decoder of the BART models,
            defaults to 4096.
        activation_function (str):
            The non-linear activation function(function or string) in the encoder and pooler of the BART models,
            defaults to :obj:`"gelu"`.
        dropout (float):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler of the BART
            model, defaults to 0.9.
        attention_dropout (float):
            The dropout ratio for the attention probabilities, defaults to 0.9.
        activation_dropout (float):
            The dropout ratio for activations inside the fully connected layer of the BART model, defaults to 0.9.
        classifier_dropout (float):
            The dropout ratio for classifier of the BART model, defaults to 0.9.
         max_position_embeddings (int):
            The maximum sequence length that this model might ever be used with. Typically set this to something
            large just in case (e.g., 512 or 1024 or 2048), defaults to 1024.
        init_std (float):
            The standard deviation of the truncated_normal_initializer forinitializing all weight matrices,
            defaults to 0.02.
        scale_embedding (bool):
            Scale embeddings by diving by sqrt(d_model), defaults to :obj:`False`.
        max_eos_token_id (int):
            The id of the token to force as the last generated token when :obj:`max_length` is reached.
            Usually set to :obj:`eos_token_id`, defaults to 2.
    """
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self,
                 vocab_size: int = 50265,
                 max_position_embeddings: int = 1024,
                 encoder_layers: int = 12,
                 encoder_ffn_dim: int = 4096,
                 encoder_attention_heads: int = 16,
                 decoder_layers: int = 12,
                 decoder_ffn_dim: int = 4096,
                 decoder_attention_heads: int = 16,
                 activation_function: str = "gelu",
                 d_model: int = 1024,
                 dropout: float = 0.9,
                 attention_dropout: float = 0.9,
                 activation_dropout: float = 0.9,
                 init_std=0.02,
                 scale_embedding: bool = False,
                 pad_token_id: int = 1,
                 bos_token_id: int = 0,
                 eos_token_id: int = 2,
                 decoder_start_token_id: int = 0,
                 max_eos_token_id: int = 2,
                 **kwargs
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.activation_function = activation_function

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout

        self.init_std = init_std
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.max_eos_token_id = max_eos_token_id
        self.use_cache = kwargs.pop("use_cache", False)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a BartConfig from the path to a JSON file of parameters.

        Args:
            json_file (Union[str, PathLike]): Path to the JSON file containing the parameters.

        Returns:
            BartConfig: The configuration object instantiated from that JSON file.
        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]) -> Dict:
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __str__(self):
        return '\n'.join(('%s:%s' % item for item in self.__dict__.items()))


ACT2FN = {
    "relu": ops.ReLU(),
    "gelu": ops.GeLU(),
    "tanh": ops.Tanh(),
    "sigmoid": ops.Sigmoid(),
}


@ops.constexpr
def generation_decoder_start(batch_size, decoder_start_token_id):
    return np.zeros([batch_size, 1], dtype=mindspore.float32) + decoder_start_token_id


@ops.constexpr
def generation_tensor(shape):
    return np.zeros(shape, dtype=mindspore.float32)


def load_bart_ckpt(ckpt, config=None):
    if config is None:
        config = BartConfig()
    bart_model = BartModel(config)
    bart_ckpt = mindspore.load_checkpoint(ckpt)
    mindspore.load_param_into_net(bart_model, bart_ckpt)
    model = BartForConditionalGeneration(bart_model, config)
    return model


def load_model_ckpt(ckpt, config=None):
    if config is None:
        config = BartConfig()
    bart_model = BartModel(config)
    model = BartForConditionalGeneration(bart_model, config)
    model_ckpt = mindspore.load_checkpoint(ckpt)
    mindspore.load_param_into_net(model, model_ckpt)
    return model


def shift_tokens_left(input_ids: mindspore.Tensor, pad_token_id: int):
    zeros = ops.Zeros()
    shifted_input_ids = zeros(input_ids.shape, mindspore.dtype.int32)
    shifted_input_ids[:, :-1] = input_ids[:, 1:]
    shifted_input_ids[:, -1] = pad_token_id
    select = ops.Select()
    cond = shifted_input_ids != -100
    replace_mat = ops.Zeros()(input_ids.shape, mindspore.dtype.int32) + pad_token_id
    shifted_input_ids = select(cond, shifted_input_ids, replace_mat)
    return shifted_input_ids


def shift_tokens(input_ids, pad_token_id):
    zeros = ops.Zeros()
    select = ops.Select()
    shifted_input_ids = zeros(input_ids.shape, mindspore.float32)
    shifted_input_ids[:, :] = input_ids
    cond = shifted_input_ids != -100
    replace_mat = ops.Zeros()(input_ids.shape, mindspore.float32) + pad_token_id
    shifted_input_ids = select(cond, shifted_input_ids, replace_mat)
    return shifted_input_ids


def _make_causal_mask(input_ids_shape: mindspore.Tensor.shape):
    """generate causal mask for padding future"""
    bsz, tgt_len = input_ids_shape
    mask = generation_tensor([tgt_len, tgt_len]) - 1e9
    mask_size = mask.shape
    mask_cond = nn.Range(0, mask_size[-1], 1)()
    mask_cond_contact = (mask_cond + 1).reshape(mask_size[-1], 1)
    mask_cond = mask_cond >= mask_cond_contact
    replace = generation_tensor(mask_cond.shape)
    select = ops.Select()
    mask = select(mask_cond, mask, replace)
    broadcast_to = ops.BroadcastTo((bsz, 1, tgt_len, tgt_len))
    return broadcast_to(mask[None, None, :, :])


def _expand_mask(mask: mindspore.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    broadcast_to = ops.BroadcastTo((bsz, 1, tgt_len, src_len))
    expanded_mask = broadcast_to(mask[:, None, None, :])
    cast = ops.Cast()
    type_dst = mindspore.float32
    select = ops.Select()
    cond = expanded_mask == 1
    inverted_mask = 1 - expanded_mask
    expanded_mask = inverted_mask * -1e9
    expanded_mask = cast(expanded_mask, type_dst)
    inverted_mask = cast(inverted_mask, type_dst)
    inverted_mask = select(cond, inverted_mask, expanded_mask)
    return inverted_mask


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def construct(self, input_ids_shape: mindspore.Tensor.shape):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        _, seq_len = input_ids_shape[:2]

        positions = nn.Range(0, seq_len)()
        return super().construct(positions + self.offset)


class BartAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 1.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super(BartAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.is_decoder = is_decoder
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

        self.dropout = nn.Dropout(keep_prob=dropout)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.batchmatmul = ops.BatchMatMul()
        self.softmax = ops.Softmax(axis=-1)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        tensor = self.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim))
        return self.transpose(tensor, (0, 2, 1, 3))

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            key_value_states: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None):
        """

        Args:
            hidden_states: the input hidden_states.
            key_value_states: the hidden_state of key_value.
            attention_mask: Input attention mask sequence.

        Returns:

        """

        bsz, tgt_len, embed_dim = hidden_states.shape

        query_states = self.q_proj(hidden_states) * self.scaling

        if key_value_states is not None:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = self.reshape(query_states, proj_shape)
        key_states = self.reshape(key_states, proj_shape)
        value_states = self.reshape(value_states, proj_shape)

        src_len = key_states.shape[1]
        key_states = self.transpose(key_states, (0, 2, 1))
        attn_weights = self.batchmatmul(query_states, key_states)

        if attention_mask is not None:
            attn_weights = self.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = self.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = self.softmax(attn_weights)
        attn_probs = self.dropout(attn_weights)

        attn_output = self.batchmatmul(attn_probs, value_states)

        attn_output = self.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = self.transpose(attn_output, (0, 2, 1, 3))
        attn_output = self.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output


class BartEncoderLayer(nn.Cell):
    """ EncoderLayer """
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout
        )
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

        self.dropout = nn.Dropout(keep_prob=config.dropout)
        self.activation_dropout = nn.Dropout(keep_prob=config.activation_dropout)

    def construct(self,
                  hidden_states: mindspore.Tensor,
                  attention_mask: mindspore.Tensor):
        """

        Args:
            hidden_states: the input hidden_states.
            attention_mask: Input attention mask sequence.
        Returns:

        """
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartDecoderLayer(nn.Cell):
    """BartDecoderLayer"""
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.fc1 = nn.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Dense(config.decoder_ffn_dim, self.embed_dim)

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

        self.dropout = nn.Dropout(keep_prob=config.dropout)
        self.activation_dropout = nn.Dropout(keep_prob=config.activation_dropout)

    def construct(self,
                  hidden_states: mindspore.Tensor,
                  attention_mask: Optional[mindspore.Tensor] = None,
                  encoder_hidden_states: Optional[mindspore.Tensor] = None,
                  encoder_attention_mask: Optional[mindspore.Tensor] = None):
        """

        Args:
            hidden_states: the input hidden_states.
            attention_mask: Input attention mask sequence.
            encoder_hidden_states: the hidden_states of encode layer.
            encoder_attention_mask: the attention_mask of encode layer.

        Returns:

        """
        residual = hidden_states

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartEncoder(nn.Cell):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig.
        embed_tokens (mindspore.nn.Embedding): output embedding.
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.encoder_layers = config.encoder_layers
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=self.padding_idx)
        self.embed_positions = BartLearnedPositionalEmbedding(
            self.max_source_positions,
            self.embed_dim,
        )
        self.layers = nn.CellList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])

        self.layernorm_embedding = nn.LayerNorm([self.embed_dim])

        self.dropout = nn.Dropout(config.dropout)

        self.shape = ops.Shape()
        self.reshape = ops.Reshape()

    def construct(self,
                  input_ids=None,
                  attention_mask=None):
        r"""

        Args:
            input_ids: Input index sequence.
            attention_mask: Input attention mask sequence.

        Returns:

        """
        input_shape = self.shape(input_ids)
        input_ids = self.reshape(input_ids, (-1, input_shape[-1]))

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask)

        for idx in range(self.encoder_layers):
            hidden_states = self.layers[idx](
                hidden_states,
                attention_mask=attention_mask,
            )

        return hidden_states


class BartDecoder(nn.Cell):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (mindspore.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_dim = config.d_model
        self.decoder_layers = config.decoder_layers
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        self.embed_positions = BartLearnedPositionalEmbedding(
            self.max_target_positions,
            self.embed_dim,
        )
        self.layers = nn.CellList([BartDecoderLayer(config) for _ in range(self.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([self.embed_dim])

        self.dropout = nn.Dropout(keep_prob=config.dropout)

        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape):
        """ generation decoder attention_mask """
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(input_shape)

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def construct(self,
                  input_ids=None,
                  attention_mask=None,
                  encoder_hidden_states=None,
                  encoder_attention_mask=None
                  ):
        r"""

        Args:
            input_ids:Input index sequence.
            attention_mask: Input attention mask sequence.
            encoder_hidden_states: the hidden_states of encode.
            encoder_attention_mask: the attention_mask of encode.

        Returns:

        """
        input_shape = self.shape(input_ids)
        input_ids = self.reshape(input_ids, (-1, input_shape[-1]))
        input_ids = self.cast(input_ids, mindspore.int32)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape)

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask = _expand_mask(encoder_attention_mask, tgt_len=input_shape[-1])

        positions = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = self.dropout(hidden_states)

        for idx in range(self.decoder_layers):
            hidden_states = self.layers[idx](
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states


class BartModel(nn.Cell):
    """BartModel"""
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.d_model
        self.decoder_start_token_id = config.decoder_start_token_id
        self.padding_idx = config.pad_token_id

        self.shared = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.padding_idx)
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

    def construct(self,
                  input_ids=None,
                  attention_mask=None,
                  decoder_input_ids=None,
                  decoder_attention_mask=None,
                  ):
        """

        Args:
            input_ids: Input index sequence.
            attention_mask: Input attention mask sequence.
            decoder_input_ids: Input index sequence for decoder.
            decoder_attention_mask: Input attention mask sequence for decoder.

        Returns:

        """

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask
        )

        return decoder_outputs


class BartForConditionalGeneration(nn.Cell):
    """BartForConditionalGeneration"""

    def __init__(self, model: BartModel, config: BartConfig):
        super(BartForConditionalGeneration, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.max_eos_token_id = config.eos_token_id

        self.model = model
        self.lm_head = nn.Dense(config.d_model, self.model.shared.vocab_size)
        self.final_bias = mindspore.Parameter(mindspore.ops.Zeros()((1, self.vocab_size), mindspore.float32),
                                              name='final_bias', requires_grad=True)

        self.reshape = ops.Reshape()
        self.expand_dim = ops.ExpandDims()
        self.concat = ops.Concat(-1)
        self.cast = ops.Cast()
        self.loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    def construct(self,
                  input_ids,
                  attention_mask=None,
                  labels=None,
                  decoder_attention_mask=None
                  ):
        """

        Args:
            input_ids: Input index sequence.
            attention_mask: Input attention mask sequence.
            labels: Label index.
            decoder_attention_mask: Input attention mask sequence for decoder.

        Returns:

        """
        decoder_input_ids = shift_tokens(labels, self.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        lm_logits = self.lm_head(outputs) + self.final_bias

        lm_loss = None

        if labels is not None:
            labels = shift_tokens_left(labels, self.pad_token_id)
            ls_logits = self.reshape(lm_logits, (-1, self.vocab_size))
            ls_labels = self.reshape(labels, (-1,))
            lm_loss = self.loss_fct(ls_logits, ls_labels)

        return lm_loss if lm_loss is not None else lm_logits

    def create_beam_search_modules(self,
                                   batch_size=55,
                                   sequence_length=64,
                                   beam_width=3):
        return BeamSearchDecoder(batch_size=batch_size,
                                 vocab_size=self.vocab_size,
                                 decoder=self.model.decoder,
                                 lm_head=self.lm_head,
                                 final_bias=self.final_bias,
                                 beam_width=beam_width,
                                 max_decode_length=sequence_length,
                                 sos_id=self.decoder_start_token_id,
                                 eos_id=self.max_eos_token_id)

    def beam_search(self,
                    input_ids,
                    attention_mask=None,
                    beam_width=3,
                    beam_search_module=None):
        """

        Args:
            input_ids: shape [batch_size, src_seq_len]a batch of input_ids by src_text tensor
            attention_mask: the mask for the attention
            beam_width: size of the last dim of beam.
            beam_search_module:

        Returns: shape [batch_size, seq_len]

        """
        batch_size = input_ids.shape[0]
        encoder_output = self.model.encoder(input_ids,
                                            attention_mask)
        if beam_width > 1:
            broadcast_state = ops.BroadcastTo((batch_size, beam_width,
                                               encoder_output.shape[1], encoder_output.shape[2]))
            broadcast_mask = ops.BroadcastTo((batch_size, beam_width,
                                              attention_mask.shape[1]))
            encoder_output = self.expand_dim(encoder_output, 1)
            encoder_output = broadcast_state(encoder_output)
            attention_mask = self.expand_dim(attention_mask, 1)
            attention_mask = broadcast_mask(attention_mask)
            encoder_output = self.reshape(encoder_output,
                                          (batch_size * beam_width, encoder_output.shape[2], encoder_output.shape[3]))
            attention_mask = self.reshape(attention_mask,
                                          (batch_size * beam_width, attention_mask.shape[2]))
        predicted_ids = beam_search_module(encoder_output, attention_mask)
        return predicted_ids


class BartForConditionalGenerationFineTuneCell(nn.Cell):
    """BartFineTuneCell"""
    def __init__(self, net, optimizer, sens=1.0):
        super(BartForConditionalGenerationFineTuneCell, self).__init__()
        self.network = net
        self.pad_token_id = net.pad_token_id
        self.optimizer = optimizer
        self.weights = mindspore.ParameterTuple(self.network.trainable_params())
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.parallel_mode = mindspore.context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = ops.composite.HyperMap()
        self.cast = ops.operations.Cast()

    def construct(self,
                  input_ids,
                  attention_mask=None,
                  labels=None):
        """Defines the computation performed."""
        weights = self.weights
        decoder_attention_mask = input_ids != self.pad_token_id
        decoder_attention_mask = self.cast(decoder_attention_mask, mindspore.int32)
        loss = self.network(input_ids,
                            attention_mask,
                            labels,
                            decoder_attention_mask)
        grads = self.grad(self.network, weights)(input_ids,
                                                 attention_mask,
                                                 labels,
                                                 decoder_attention_mask,
                                                 self.cast(ops.functional.tuple_to_array((self.sens,)),
                                                           mindspore.dtype.float32))
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
