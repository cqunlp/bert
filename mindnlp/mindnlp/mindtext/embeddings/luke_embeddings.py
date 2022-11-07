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
    luke embeddings
"""
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype


class EntityEmbeddings(nn.Cell):
    """entity embeddings for luke model"""

    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()

        self.entity_emb_size = config.entity_emb_size
        self.hidden_size = config.hidden_size
        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)

        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Dense(config.entity_emb_size, config.hidden_size, has_bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.unsqueezee = ops.ExpandDims()

    def construct(self, entity_ids, position_ids, token_type_ids=None):
        """EntityEmbeddings for luke"""

        if token_type_ids is None:
            token_type_ids = ops.zeros_like(entity_ids)
        # print("position_ids:", position_ids.shape)
        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.entity_emb_size != self.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)
        entity_position_ids_int = clamp(position_ids)

        position_embeddings = self.position_embeddings(entity_position_ids_int)
        # print("0.position_embeddings:", position_embeddings.shape)

        position_ids = position_ids.astype(mstype.int32)
        position_embedding_mask = 1.0 * self.unsqueezee((position_ids != -1), -1)
        # print("1.position_embeddings:", position_embeddings.shape)

        position_embeddings = position_embeddings * position_embedding_mask
        # print("2.position_embeddings:", position_embeddings.shape)

        position_embeddings = ops.reduce_sum(position_embeddings, -2)
        # print("3.position_embeddings:", position_embeddings.shape)

        position_embeddings = position_embeddings / clamp(ops.reduce_sum(position_embedding_mask, -2), minimum=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # print("entity_embeddings:", entity_embeddings.shape)
        # print("4.position_embeddings:", position_embeddings.shape)
        # print("token_type_embeddings:", token_type_embeddings.shape)

        embeddings = entity_embeddings + position_embeddings
        embeddings += token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def clamp(x, minimum=0):
    mask = x > minimum
    x = x * mask + minimum
    return x


class RobertaEmbeddings(nn.Cell):
    """
    RoBERTaEmbeddings
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id
                                            )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.add = ops.Add()
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.layer_norm = nn.LayerNorm([config.hidden_size],
                                       epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size,
                                                padding_idx=self.padding_idx)

    def construct(self,
                  input_ids=None,
                  token_type_ids=None,
                  position_ids=None,
                  inputs_embeds=None,
                  past_key_values_length=0):
        """
        Returns the result of the model after loading the pre-training weights

        Args:
            input_ids:A vector containing the transformation of characters into corresponding ids.
            token_type_ids:A vector containing segemnt ids.
            position_ids:A vector containing position_ids.
            inputs_embeds:A vector containing embeds
            past_key_values_length: A number

        Returns:
            RoBERTa embeddings
        """
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = create_position_ids_from_input_ids(inputs_embeds)

        input_shape = input_ids.shape
        # seq_length = input_shape[1]
        if token_type_ids is None:
            token_type_ids = ops.Zeros(input_shape, dtype=mstype.int64)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = self.add(inputs_embeds, token_type_embeddings)
        position_ids = position_ids.astype(mstype.int32)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.add(embeddings, position_embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def create_position_ids_from_input_ids(input_ids, padding_idx=0, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
       input_ids:
       padding_idx:
       past_key_values_length:
    Returns:
        Tensor
    """
    mask = input_ids != padding_idx
    mask = (1.0 * mask)

    cumsum = ops.CumSum()
    incremental_indices = (cumsum(mask, 1) + past_key_values_length) * mask
    return incremental_indices + padding_idx
