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
    Encoder classes for XLNet.
"""

import os
import json
from typing import Union, Dict, List, Tuple, Optional

from mindspore.ops import constexpr
import mindspore.numpy as np

import mindspore
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Normal
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore._checkparam import Validator
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = mindspore.ops.MultitypeFuncGraph("grad_scale")
reciprocal = mindspore.ops.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


_grad_overflow = mindspore.ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = mindspore.ops.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


clip_grad = mindspore.ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = mindspore.ops.dtype(grad)
    if clip_type == 0:
        new_grad = mindspore.ops.clip_by_value(grad,
                                               mindspore.ops.cast(mindspore.ops.tuple_to_array((-clip_value,)), dt),
                                               mindspore.ops.cast(mindspore.ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, mindspore.ops.cast(mindspore.ops.tuple_to_array((clip_value,)), dt))
    return new_grad


@constexpr
def generate_arange_tensor(beg, end, st):
    return np.arange(beg, end, st)


ACT2FN = {"gelu": nn.GELU(), "relu": mindspore.ops.ReLU()}


class XLNetConfig:
    """
    This is the configuration class to store the configuration of XLNetModel.The class can be instantiated
    from a json file.

    Args:
        vocab_size (int): Vocabulary size of the XLNet model. Defines the number of different tokens that can be
            represented, defaults to 32000.
        d_model (int): Dimensionality of the encoder layers and the pooler layer, defaults to 1024.
        n_layer (int): Number of hidden layers in the Transformer encoder, defaults to 24.
        n_head (int): Number of attention heads for each attention layer, defaults to 16.
        d_inner (int): Dimensionality of the feed-forward layer in the Transformer encoder, defaults to 4096.
        ff_activation (str): The non-linear activation function (function or string), defaults to "gelu".
        untie_r (bool): Whether or not to untie relative position biases, defaults to True.
        attn_type (str): The attention type used by the model, defaults to "bi".
        initializer_range (float): The standard deviation of the truncated_normal_initializer for initializing all
            weight matrices, defaults to 0.02.
        layer_norm_eps (float): The epsilon used by the layer normalization layers, defaults to 1e-12.
        dropout (float): The dropout probability for all fully connected layers in the embeddings, encoder,
            and pooler, defaults to 0.1.
        mem_len (int): The number of tokens to cache. The key/value pairs that have already been pre-computed
            in a previous forward pass won't be re-computed, defaults to 512.
        reuse_len (int, Optional): The number of tokens in the current batch to be cached and reused in the future,
            defaults to None.
        use_mems (bool): Whether or not the model should make use of the recurrent memory mechanism, defaults to False.
        bi_data (bool): Whether or not to use bidirectional input pipeline. Usually set to True during pretraining and
            False during finetuning, defaults to False.
        clamp_len (int): Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means
            no clamping, defaults to -1.
        same_length (bool): Whether or not to use the same attention length for each token, defaults to False.
        pad_token_id (int): The padding index in inputs, defaults to 5.
        bos_token_id (int): The begin index in inputs, defaults to 1.
        eos_token_id (int): The end index in inputs, defaults to 2.

    Examples:
        >>> xlconfig = XLNetConfig.from_json_file('json_file_path')
    """

    model_type = "xlnet"
    keys_to_ignore_at_inference = ["mems"]

    def __init__(
            self,
            vocab_size: int = 32000,
            d_model: int = 1024,
            n_layer: int = 24,
            n_head: int = 16,
            d_inner: int = 4096,
            ff_activation: str = "gelu",
            untie_r: bool = True,
            attn_type_bi: bool = True,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            dropout: float = 0.1,
            mem_len: int = 512,
            reuse_len: Optional[int] = None,
            use_mems: bool = False,
            bi_data: bool = False,
            clamp_len: int = -1,
            same_length: bool = False,
            pad_token_id: int = 5,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        assert d_model % n_head == 0
        if "d_head" in kwargs:
            assert (kwargs["d_head"] == d_model // n_head), \
                f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type_bi = attn_type_bi

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.use_mems = use_mems
        self.return_dict = kwargs.pop("return_dict", True)
        self.architectures = kwargs.pop("architectures", None)
        self.model_type = kwargs.pop("model_type", "xlnet")
        self.use_inputs_embeds = kwargs.pop("use_inputs_embeds", False)
        self.use_perm_mask = kwargs.pop("use_perm_mask", False)
        self.use_target_mapping = kwargs.pop("use_target_mapping", False)
        self.use_input_mask = kwargs.pop("use_input_mask", False)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a XLNetConfig from the path to a JSON file of parameters.

        Args:
            json_file (Union[str, PathLike]): Path to the JSON file containing the parameters.

        Returns:
            XLNetConfig: The configuration object instantiated from that JSON file.
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


class XLNetRelativeAttention(nn.Cell):
    """
    XLNet relative attention.

    Args:
        config (XLNetConfig): XLNetConfig object.

    Examples:
        >>> xlconfig = XLNetConfig.from_json_file('json_file_path')
        >>> xl_rel_attn = XLNetRelativeAttention(xlconfig)
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetRelativeAttention, self).__init__()
        self.n_head = Validator.check_positive_int(config.n_head)
        self.d_head = Validator.check_positive_int(config.d_head)
        self.d_model = Validator.check_positive_int(config.d_model)
        Validator.check_value_type('layer_norm_eps', config.layer_norm_eps, [float])
        Validator.check_value_type('dropout', config.dropout, [float])
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.d_model, self.n_head, self.d_head]),
            name='attention_q')
        self.k = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.d_model, self.n_head, self.d_head]),
            name='attention_k')
        self.v = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.d_model, self.n_head, self.d_head]),
            name='attention_v')
        self.o = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.d_model, self.n_head, self.d_head]),
            name='attention_o')
        self.r = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.d_model, self.n_head, self.d_head]),
            name='attention_r')

        self.r_r_bias = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.n_head, self.d_head]),
            name='attention_r_r_bias')
        self.r_s_bias = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.n_head, self.d_head]),
            name='attention_r_s_bias')
        self.r_w_bias = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [self.n_head, self.d_head]),
            name='attention_r_w_bias')
        self.seg_embed = Parameter(
            initializer(Normal(mean=0.0, sigma=config.initializer_range), [2, self.n_head, self.d_head]),
            name='attention_seg_embed')

        self.layer_norm = nn.LayerNorm([self.d_model], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(1 - config.dropout)

    @staticmethod
    def rel_shift(x: Tensor, klen: int = -1) -> Tensor:
        """
        Perform relative shift to form the relative attention score.

        Args:
            x (Tensor): The tensor will be shifted.
            klen (int): The length of shifted tensor.

        Returns:
            Tensor: The shifted tensor.
        """
        x_size = x.shape
        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        x = x[:, 0:klen, :, :]
        return x

    @staticmethod
    def rel_shift_bnij(x: Tensor, klen: int = -1) -> Tensor:
        """
        Perform relative shift to form the relative attention score.

        Args:
            x (Tensor): The tensor will be shifted.
            klen (int): The length of shifted tensor.

        Returns:
            Tensor: The shifted tensor.
        """
        x_size = x.shape
        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = x[:, :, :, :klen]
        return x

    def rel_attn_core(
            self,
            q_head: Tensor,
            k_head_h: Tensor,
            v_head_h: Tensor,
            k_head_r: Tensor,
            seg_mat: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Core relative positional attention operations.

        Args:
            q_head (Tensor): Content-stream query head.
            k_head_h (Tensor): Content-based key head.
            v_head_h (Tensor): Content-based value head.
            k_head_r (Tensor): Position-based key head.
            seg_mat (Tensor, Optional): Segment embedding.
            attn_mask (Tensor, Optional): Attention mask.

        Returns:
            Tensor: Relative attention score.
        """
        # Content based attention score.
        q_r_w = q_head + self.r_w_bias
        q_r_w = mindspore.ops.transpose(q_r_w, (2, 1, 0, 3))
        k_head_h = mindspore.ops.transpose(k_head_h, (2, 1, 3, 0))
        ac = mindspore.ops.matmul(q_r_w, k_head_h)
        ac = mindspore.ops.transpose(ac, (1, 0, 2, 3))

        # Position based attention score.
        q_r_r = q_head + self.r_r_bias
        q_r_r = mindspore.ops.transpose(q_r_r, (2, 1, 0, 3))
        k_head_r = mindspore.ops.transpose(k_head_r, (2, 1, 3, 0))
        bd = mindspore.ops.matmul(q_r_r, k_head_r)
        bd = mindspore.ops.transpose(bd, (1, 0, 2, 3))
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # Segment based attention score.
        if not isinstance(seg_mat, Tensor):
            ef = 0
        else:
            q_r_s = q_head + self.r_s_bias
            i, b, n, d = q_r_s.shape
            s, _, _ = self.seg_embed.shape
            q_r_s = mindspore.ops.transpose(q_r_s.reshape((i * b, n, d)), (1, 0, 2))
            seg_embed = mindspore.ops.transpose(self.seg_embed, (1, 2, 0))
            ef = mindspore.ops.transpose(mindspore.ops.matmul(q_r_s, seg_embed).reshape((n, i, b, s)), (1, 2, 0, 3))

            seg_mat = mindspore.ops.transpose(seg_mat, (2, 0, 1, 3))
            ef = mindspore.ops.transpose(ef, (1, 0, 3, 2))

            ef = mindspore.ops.matmul(seg_mat, ef)
            ef = mindspore.ops.transpose(ef, (0, 3, 1, 2))

        # Merge attention scores and perform masking.
        attn_score = (ac + bd + ef) * self.scale
        if isinstance(attn_mask, Tensor):
            if attn_mask.dtype == mstype.float16:
                attn_score = attn_score - 65500 * mindspore.ops.transpose(attn_mask, (2, 3, 0, 1))
            else:
                attn_score = attn_score - 1e30 * mindspore.ops.transpose(attn_mask, (2, 3, 0, 1))

        # Attention probability.
        softmax = nn.Softmax(axis=3)
        attn_prob = softmax(attn_score)
        attn_prob = self.dropout(attn_prob)

        # Attention output.
        attn_vec = mindspore.ops.matmul(attn_prob, mindspore.ops.transpose(v_head_h, (1, 2, 0, 3)))
        attn_vec = mindspore.ops.transpose(attn_vec, (2, 0, 1, 3))

        return attn_vec

    def post_attention(self, h: Tensor, attn_vec: Tensor, residual: bool = True) -> Tensor:
        """
        Post-attention processing.

        Args:
            h (Tensor): H hidden states.
            attn_vec (Tensor): Attention vector.
            residual (bool): Residual connections.

        Returns:
            Tensor: Post-attention.
        """
        # Post-attention projection (back to `d_model`).
        i_, b_, n_, d_ = attn_vec.shape
        h_, _, _ = self.o.shape
        attn_vec = attn_vec.reshape((i_, b_, n_ * d_))
        o = mindspore.ops.transpose(self.o.reshape((h_, n_ * d_)), (1, 0))
        attn_out = mindspore.ops.matmul(attn_vec, o)

        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def construct(self,
                  h: Tensor,
                  g: Tensor,
                  attn_mask_h: Tensor,
                  attn_mask_g: Tensor,
                  r: Tensor,
                  seg_mat: Tensor,
                  mems: Optional[Tensor] = None,
                  target_mapping: Optional[Tensor] = None,
                  ) -> Union[Tensor, Tuple[Tensor]]:
        """
        XLNet relative attention layer forward propagation.

        Args:
            h (Tensor): H hidden states.
            g (Tensor): G hidden states.
            attn_mask_h (Tensor): H hidden states attention mask.
            attn_mask_g (Tensor): G hidden states attention mask.
            r (Tensor): Positional encoding.
            seg_mat (Tensor): Segment embedding.
            mems (Tensor, Optional): Mems tensor.
            target_mapping (Tensor, Optional): XLNet target mapping.

        Returns:
            Union[Tensor, Tuple[Tensor]]: Relative attentions.
        """
        if isinstance(g, Tensor):
            # Two-stream attention with relative positional encoding.
            # Content based attention score.
            if isinstance(mems, Tensor) and mems.dim() > 1:
                cat_ops = mindspore.ops.Concat(axis=0)
                cat = cat_ops((mems, h))
            else:
                cat = h

            # Content-based key head.
            i_, b_, h_ = cat.shape
            _, n_, d_ = self.k.shape
            k_head_h = mindspore.ops.matmul(cat, self.k.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

            # Content-based value head.
            i_, b_, h_ = cat.shape
            _, n_, d_ = self.v.shape
            v_head_h = mindspore.ops.matmul(cat, self.v.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

            # Position-based key head.
            i_, b_, h_ = r.shape
            _, n_, d_ = self.r.shape
            k_head_r = mindspore.ops.matmul(r, self.r.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

            # Content-stream query head(h-stream).
            i_, b_, h_ = h.shape
            _, n_, d_ = self.q.shape
            q_head_h = mindspore.ops.matmul(h, self.q.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

            # Core attention ops.
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
            )

            # Post processing.
            output_h = self.post_attention(h, attn_vec_h)

            # Query-stream query head(g-stream).
            i_, b_, h_ = g.shape
            _, n_, d_ = self.q.shape
            q_head_g = mindspore.ops.matmul(g, self.q.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

            # Core attention ops.
            if isinstance(target_mapping, Tensor):
                m_, b_, n_, d_ = q_head_g.shape
                _, l_, _ = target_mapping.shape
                q_head_g = mindspore.ops.transpose(q_head_g, (1, 2, 3, 0)).reshape((b_, n_ * d_, m_))
                target_mapping = mindspore.ops.transpose(target_mapping, (2, 0, 1))
                q_head_g = mindspore.ops.matmul(q_head_g, target_mapping).reshape((b_, n_, d_, l_))
                q_head_g = mindspore.ops.transpose(q_head_g, (3, 0, 1, 2))

                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                )

                l_, b_, n_, d_ = attn_vec_g.shape
                m_, _, _ = target_mapping.shape
                attn_vec_g = mindspore.ops.transpose(attn_vec_g, (1, 2, 3, 0)).reshape((b_, n_ * d_, l_))
                target_mapping = mindspore.ops.transpose(target_mapping, (2, 1, 0))
                attn_vec_g = mindspore.ops.matmul(attn_vec_g, target_mapping).reshape((b_, n_, d_, m_))
                attn_vec_g = mindspore.ops.transpose(attn_vec_g, (3, 0, 1, 2))
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                )

            # Post processing.
            output_g = self.post_attention(g, attn_vec_g)
            return output_h, output_g

        # Multi-head attention with relative positional encoding.
        if isinstance(mems, Tensor) and mems.dim() > 1:
            cat_ops = mindspore.ops.Concat(axis=0)
            cat = cat_ops((mems, h))
        else:
            cat = h

        # Content heads.
        i_, b_, h_ = h.shape
        _, n_, d_ = self.q.shape
        q_head_h = mindspore.ops.matmul(h, self.q.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

        i_, b_, h_ = cat.shape
        _, n_, d_ = self.k.shape
        k_head_h = mindspore.ops.matmul(cat, self.k.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

        i_, b_, h_ = cat.shape
        _, n_, d_ = self.v.shape
        v_head_h = mindspore.ops.matmul(cat, self.v.reshape((h_, n_ * d_))).reshape((i_, b_, n_, d_))

        # Positional heads.
        # Type casting for fp16 support.
        i_, b_, h_ = r.shape
        _, n_, d_ = self.r.shape
        k_head_r = mindspore.ops.matmul(r.astype(self.r.dtype), self.r.reshape((h_, n_ * d_))).reshape(
            (i_, b_, n_, d_))

        # Core attention ops.
        attn_vec = self.rel_attn_core(
            q_head_h,
            k_head_h,
            v_head_h,
            k_head_r,
            seg_mat=seg_mat,
            attn_mask=attn_mask_h,
        )

        # Post processing.
        output_h = self.post_attention(h, attn_vec)

        return output_h


class XLNetFeedForward(nn.Cell):
    """
    XLNet feed-forward layer in transformer encoder.

    Args:
        config (XLNetConfig): XLNetConfig object.

    Examples:
        >>> xlconfig = XLNetConfig.from_json_file('json_file_path')
        >>> xl_feed_forward = XLNetFeedForward(xlconfig)
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm([config.d_model], epsilon=config.layer_norm_eps)
        self.layer_1 = nn.Dense(config.d_model, config.d_inner)
        self.layer_2 = nn.Dense(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(1 - config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def construct(self, inp: Tensor) -> Tensor:
        """
        XLNet feed-forward layer forward propagation.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Cell):
    """
    XLNet Encoder layer.

    Args:
        config (XLNetConfig): XLNetConfig object.

    Examples:
        >>> xlconfig = XLNetConfig.from_json_file('json_file_path')
        >>> xl_layer = XLNetLayer(xlconfig)
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetLayer, self).__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(1 - config.dropout)
        self.seq_len_dim = 1

    def construct(
            self,
            output_h: Tensor,
            output_g: Tensor,
            attn_mask_h: Tensor,
            attn_mask_g: Tensor,
            r: Tensor,
            seg_mat: Tensor,
            mems: Optional[Tensor] = None,
            target_mapping: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        XLNet encoder layer forward propagation.

        Args:
            output_h (Tensor): H hidden states.
            output_g (Tensor): G hidden states.
            attn_mask_h (Tensor): H hidden states attention mask.
            attn_mask_g (Tensor): G hidden states attention mask.
            r (Tensor): Positional encoding.
            seg_mat (Tensor): Segment embedding.
            mems (Tensor, Optional): Mems tensor.
            target_mapping (Tensor, Optional): XLNet target mapping.

        Returns:
            Union[Tensor, Tuple[Tensor]]: XLNet encoder attentions.
        """
        if isinstance(output_g, Tensor):
            output_h, output_g = self.rel_attn(
                output_h,
                output_g,
                attn_mask_h,
                attn_mask_g,
                r,
                seg_mat,
                mems=mems,
                target_mapping=target_mapping,
            )
            output_h = self.ff(output_h)
            output_g = self.ff(output_g)
            return output_h, output_g

        output_h = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
        )
        output_h = self.ff(output_h)
        return output_h

    def ff_chunk(self, output_x):
        output_x = self.ff(output_x)
        return output_x


class XLNetModel(nn.Cell):
    """
    XLNet model.

    Args:
        config (XLNetConfig): XLNetConfig object.

    Examples:
        >>> xlconfig = XLNetConfig.from_json_file('json_file_path')
        >>> xlnet = XLNetModel(xlconfig)
    """

    def __init__(self, config: XLNetConfig):
        super(XLNetModel, self).__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = Parameter(initializer(Normal(mean=0.0, sigma=config.initializer_range), [1, 1, config.d_model]),
                                  name='mask_emb')
        self.layer = nn.CellList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(1 - config.dropout)
        self.cast = mindspore.ops.Cast()
        self.config = config
        self.return_dict = config.return_dict
        self.use_mems = config.use_mems
        self.attn_type_bi = config.attn_type_bi
        if not self.attn_type_bi:
            self.same_length = config.same_length
        self.bi_data = config.bi_data
        self.clamp_len = Tensor(config.clamp_len)
        self.d_model = config.d_model
        self.n_layer = config.n_layer
        self.one_hot_on_value = mindspore.Tensor(1.0, mindspore.dtype.float32)
        self.one_hot_off_value = mindspore.Tensor(0.0, mindspore.dtype.float32)

    def create_mask(self, qlen: int, mlen: int) -> Tensor:
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen (int): Sequence length.
            mlen (int): Mask length.

        Returns:
            Tensor: Mask matrix.

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        ones = mindspore.ops.Ones()
        triu = mindspore.nn.Triu()
        zeros = mindspore.ops.Zeros()
        cat = mindspore.ops.Concat(1)
        tril = mindspore.nn.Tril()

        attn_mask = ones((qlen, qlen))
        mask_up = triu(attn_mask, 1)
        attn_mask_pad = zeros((qlen, mlen))
        ret = cat((attn_mask_pad, mask_up))
        if self.same_length:
            mask_lo = tril(attn_mask, -1)
            ret = cat((ret[:, :qlen] + mask_lo, ret[:, qlen:]))

        return ret

    @staticmethod
    def positional_embedding(pos_seq: Tensor, inv_freq: Tensor, bsz: Optional[int] = None) -> Tensor:
        """
        Positional embedding.

        Args:
            pos_seq (Tensor): Positional tensor sequence.
            inv_freq (Tensor): Frequency scaling tensor.
            bsz (int, Optional): Batch size.

        Returns:
            Tensor: Positional embedding.
        """
        expand_dims = mindspore.ops.ExpandDims()
        pos_seq = expand_dims(pos_seq, 1)
        inv_freq = expand_dims(inv_freq, 0)
        sinusoid_inp = mindspore.ops.matmul(pos_seq, inv_freq)

        cat = mindspore.ops.Concat(axis=-1)
        sin = mindspore.ops.Sin()
        cos = mindspore.ops.Cos()
        pos_emb = cat((sin(sinusoid_inp), cos(sinusoid_inp)))
        pos_emb = pos_emb[:, None, :]

        if isinstance(bsz, int):
            broadcast_to = mindspore.ops.BroadcastTo((-1, bsz, -1))
            pos_emb = broadcast_to(pos_emb)

        return pos_emb

    def relative_positional_encoding(self, qlen: int, klen: int, bsz: Optional[int] = None) -> Tensor:
        """
        Relative positional encoding.

        Args:
            qlen (int): Sequence length.
            klen (int): Sequence length add number of tokens to cache.
            bsz (int, Optional): Batch size.

        Returns:
            Tensor: Relative positional encoding.
        """
        # Create relative positional encoding.
        freq_seq = generate_arange_tensor(0, self.d_model, 2.0)
        pow_ops = mindspore.ops.Pow()
        inv_freq = 1 / pow_ops(10000, (freq_seq / self.d_model))

        if self.attn_type_bi:
            beg, end = klen, -qlen
        else:
            beg, end = klen, -1

        if self.bi_data:
            fwd_pos_seq = generate_arange_tensor(beg, end, -1.0)
            bwd_pos_seq = generate_arange_tensor(-beg, -end, 1.0)

            if self.clamp_len > 0:
                fwd_pos_seq = mindspore.ops.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
                bwd_pos_seq = mindspore.ops.clip_by_value(bwd_pos_seq, -self.clamp_len, self.clamp_len)

            if isinstance(bsz, int):
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            cat = mindspore.ops.Concat(axis=1)
            pos_emb = cat((fwd_pos_emb, bwd_pos_emb))
        else:
            fwd_pos_seq = generate_arange_tensor(beg, end, -1.0)
            if self.clamp_len > 0:
                fwd_pos_seq = mindspore.ops.clip_by_value(fwd_pos_seq, -self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def cache_mem(self, curr_out: Tensor, prev_mem: Tensor) -> Tensor:
        """
        Cache hidden states into memory.

        Args:
            curr_out (Tensor): Current hidden states.
            prev_mem (Tensor): Previous memory.

        Returns:
            Tensor: Current memory.
        """
        if isinstance(self.reuse_len, int) and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if not self.mem_len or self.mem_len == 0:
            cutoff = 0
        else:
            cutoff = -self.mem_len
        if not prev_mem:
            new_mem = curr_out[cutoff:]
        else:
            cat = mindspore.ops.Concat(axis=0)
            new_mem = cat((prev_mem, curr_out))[cutoff:]

        return new_mem

    def construct(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            token_type_ids: Tensor,
            mems: Optional[List[Tensor]] = None,
            perm_mask: Optional[Tensor] = None,
            target_mapping: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor]]:
        """
        XLNet model forward propagation.

        Args:
            input_ids (Tensor): Input index sequence.
            attention_mask (Tensor): Input index attention mask.
            token_type_ids: Input index sequence type.
            mems (List[Tensor], Optional): Cache memory.
            perm_mask (Tensor, Optional): Perm mask.
            target_mapping (Tensor, Optional): Target mapping.

        Returns:
            Union[Tensor, Tuple[Tensor]]: The outputs contain `output, new_mems` or `output`.
        """
        transpose = mindspore.ops.Transpose()
        input_ids = transpose(input_ids, (1, 0))
        qlen, bsz = input_ids.shape[0], input_ids.shape[1]

        transpose = mindspore.ops.Transpose()
        token_type_ids = transpose(token_type_ids, (1, 0))
        attention_mask = transpose(attention_mask, (1, 0))

        mlen = mems[0].shape[0] if self.use_mems else 0
        klen = mlen + qlen

        # Attention mask.
        # Causal attention mask.s
        if self.attn_type_bi:
            attn_mask = None
        else:
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]

        # If input_mask is None and attention_mask is not None:
        input_mask = 1.0 - attention_mask
        if isinstance(perm_mask, Tensor):
            data_mask = input_mask[None] + perm_mask
        else:
            data_mask = input_mask[None]

        # All mems can be attended to.
        if mlen > 0:
            zeros = mindspore.ops.Zeros()
            cat = mindspore.ops.Concat(1)
            mems_mask = zeros((data_mask.shape[0], mlen, bsz))
            data_mask = cat((mems_mask, data_mask))

        if not isinstance(attn_mask, Tensor):
            attn_mask = data_mask[:, :, :, None]
        else:
            attn_mask += data_mask[:, :, :, None]

        attn_mask = (attn_mask > 0)
        attn_mask = self.cast(attn_mask, mstype.int64)

        eye = mindspore.ops.Eye()
        non_tgt_mask = -eye(qlen, qlen, mstype.float32)
        if mlen > 0:
            zeros = mindspore.ops.Zeros()
            cat = mindspore.ops.Concat(axis=-1)
            non_tgt_mask = cat((zeros((qlen, mlen)), non_tgt_mask))
        non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0)
        non_tgt_mask = self.cast(non_tgt_mask, mstype.int64)

        # Word embeddings and prepare h & g hidden states.
        word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if isinstance(target_mapping, Tensor):
            broadcast_to = mindspore.ops.BroadcastTo((target_mapping.shape[0], bsz, -1))
            word_emb_q = broadcast_to(self.mask_emb)
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None

        # Segment embedding.
        if isinstance(token_type_ids, Tensor):
            # Convert `token_type_ids` to one-hot `seg_mat`.
            if mlen > 0:
                cat = mindspore.ops.Concat(axis=0)
                mem_pad = mindspore.ops.Zeros((mlen, bsz), mstype.int32)
                cat_ids = cat((mem_pad, token_type_ids))
            else:
                cat_ids = token_type_ids

            # `1` indicates not in the same segment [qlen x klen x bsz].
            seg_mat = self.cast((token_type_ids[:, None] != cat_ids[None, :]), mstype.int64)
            one_hot = mindspore.ops.OneHot()
            seg_mat = one_hot(seg_mat, 2, self.one_hot_on_value, self.one_hot_off_value)
        else:
            seg_mat = None

        # Positional encoding.
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        new_mems = ()
        if not isinstance(mems, Tensor):
            mems = [None] * len(self.layer)

        for i, layer_module in enumerate(self.layer):
            if self.use_mems:
                # Cache new mems.
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)

            if isinstance(output_g, Tensor):
                output_h, output_g = layer_module(
                    output_h,
                    output_g,
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    r=pos_emb,
                    seg_mat=seg_mat,
                    mems=mems[i],
                    target_mapping=target_mapping,
                )
            else:
                output_h = layer_module(
                    output_h,
                    output_g,
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    r=pos_emb,
                    seg_mat=seg_mat,
                    mems=mems[i],
                    target_mapping=target_mapping,
                )

        output = self.dropout(output_g if isinstance(output_g, Tensor) else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method).
        transpose = mindspore.ops.Transpose()
        output = transpose(output, (1, 0, 2))

        if not self.use_mems:
            return output

        return output, new_mems


class XLNetFinetuneCell(nn.Cell):
    """
    XLNet finetune cell.

    Args:
        network (nn.Cell): XLNet model, such as XLNetForClassification.
        optimizer (nn.Cell): Optimizer.
        scale_update_cell (nn.Cell, Optional): Scaling loss.
    """

    def __init__(self, network: nn.Cell, optimizer: nn.Optimizer, scale_update_cell: Optional[nn.Cell] = None):
        super(XLNetFinetuneCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = mindspore.ops.GradOperation(get_by_list=True,
                                                sens_param=True)
        self.reducer_flag = False
        self.allreduce = mindspore.ops.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = mindspore.ops.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = mindspore.ops.FloatStatus()
            self.addn = mindspore.ops.AddN()
            self.reshape = mindspore.ops.Reshape()
        else:
            self.alloc_status = mindspore.ops.NPUAllocFloatStatus()
            self.get_status = mindspore.ops.NPUGetFloatStatus()
            self.clear_status = mindspore.ops.NPUClearFloatStatus()
        self.reduce_sum = mindspore.ops.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = mindspore.ops.LessEqual()
        self.hyper_map = mindspore.ops.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids: Tensor,
                  token_type_id: Tensor,
                  attention_mask: Tensor,
                  label: Tensor,
                  sens: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        XLNet Finetune.

        Args:
            input_ids (Tensor): Input index sequence.
            token_type_id (Tensor): Input type index sequence.
            attention_mask (Tensor): Input attention mask sequence.
            label (Tensor): Label index.
            sens (int, Optional): Sensitivity about gradient with respect to output.

        Returns：
            Tuple[Tensor, Tensor]： Return loss.
        """

        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            attention_mask,
                            token_type_id,
                            label)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = mindspore.ops.tuple_to_array((sens,))

        if not self.gpu_target:
            init = self.alloc_status()
            init = mindspore.ops.depend(init, loss)
            clear_status = self.clear_status(init)
            scaling_sens = mindspore.ops.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(input_ids,
                                                 attention_mask,
                                                 token_type_id,
                                                 label,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(mindspore.ops.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(mindspore.ops.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            init = mindspore.ops.depend(init, grads)
            get_status = self.get_status(init)
            init = mindspore.ops.depend(init, get_status)
            flag_sum = self.reduce_sum(init, (0,))
        else:
            flag_sum = self.hyper_map(mindspore.ops.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return loss, cond


class XLNetForClassification(nn.Cell):
    """
    XLNet classification finetune cell.

    Args:
        model (nn.Cell): XLNet model.
        config (XLNetConfig): XLNet model config.
        num_class (num_class): The number of class.
        loss (nn.Cell, Optional): Loss function.

    Returns:
        Tensor: Loss with gradient.
    """

    def __init__(self, model: XLNetModel, config: XLNetConfig, num_class: int = 2,
                 loss: Optional[nn.Cell] = None) -> Tensor:
        super(XLNetForClassification, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(1 - config.dropout)
        self.classifier = nn.Dense(config.d_model, num_class)
        if isinstance(loss, nn.Cell):
            self.loss = loss
        else:
            self.loss = mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, input_ids, attention_mask, token_type_ids, label: Optional[Tensor] = None):
        """
        XLNet classification model forward propagation.

        Args:
            input_ids (Tensor): Input index sequence.
            attention_mask (Tensor): Input attention mask sequence.
            token_type_ids (Tensor): Input type index sequence.
            label (Tensor): Label index.

        Returns：
            Tuple[Tensor, Tensor]： Return loss.
        """
        output = self.model(input_ids, attention_mask, token_type_ids)
        mean = mindspore.ops.ReduceMean(keep_dims=False)
        pool_output = mean(output, 1)
        pool_output = self.dropout(pool_output)
        output = self.classifier(pool_output)
        if self.training:
            squeeze = mindspore.ops.Squeeze(1)
            label = squeeze(label)
            loss = self.loss(output, label)
        else:
            loss = output
        return loss
