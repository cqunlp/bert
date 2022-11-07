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
"""DPCNN model."""
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import ParameterTuple
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import GradOperation
from mindspore.communication import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from ...common.utils.config import Config
from ...embeddings.static_embedding import StaticEmbedding
from ...modules.encoder.conv import ConvEncoder
from ...modules.decoder.norm_decoder import NormalDecoder

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


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
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class DPCNN(nn.Cell):
    """
    DPCNN model.

    Args:
        init_embed(StaticEmbedding): the StaticEmbedding constructed by the vocabulary of dataset.
        config(Config): Model configuration.

    Returns:
        Tensor. The running result of DPCNN model.
    """

    def __init__(self, init_embed: StaticEmbedding, config: Config):
        super(DPCNN, self).__init__()
        self.encoder = ConvEncoder(init_embed, config.model.encoder.num_filters, config.model.encoder.kernel_size,
                                   config.model.encoder.num_layers, config.model.encoder.embed_dropout)
        self.decoder = NormalDecoder(config.model.decoder.num_filters, config.model.decoder.num_classes,
                                     config.model.decoder.classes_dropout)
        self.cast = P.Cast()

    def construct(self, words):
        x = self.encoder(words)
        x = self.decoder(x)
        x = self.cast(x, mstype.float32)
        return x


class DPCNNNetWithLoss(nn.Cell):
    """
    Provide DPCNN training loss

    Args:
        net(nn.Cell): DPCNN model.
        loss(Loss): DPCNN loss.
    """

    def __init__(self, net, loss):
        super(DPCNNNetWithLoss, self).__init__()
        self.dpcnn = net
        self.loss_func = loss
        self.squeeze = P.Squeeze(axis=1)
        self.print = P.Print()

    def construct(self, src_tokens, label_idx):
        predict_score = self.dpcnn(src_tokens)
        label_idx = self.squeeze(label_idx)
        predict_score = self.loss_func(predict_score, label_idx)
        return predict_score


class DPCNNTrainOneStep(nn.Cell):
    """
    DPCNN train class.

    Args:
        net(nn.Cell):
        loss(Loss): DPCNN loss.
        optimizer(Optimizer): DPCNN optimizer.
    """
    def __init__(self, net, loss, optimizer, sens=1.0):
        super(DPCNNTrainOneStep, self).__init__(auto_prefix=False)
        self.network = DPCNNNetWithLoss(net, loss)
        self.network.init_parameters_data()
        self.optimizer = optimizer
        self.weights = ParameterTuple(self.network.trainable_params())
        self.grad = GradOperation(get_by_list=True, sens_param=True)
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

    def construct(self, src_token_text, label_idx_tag):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(src_token_text, label_idx_tag)
        grads = self.grad(self.network, weights)(src_token_text,
                                                 label_idx_tag,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


class DPCNNInferCell(nn.Cell):
    """
    Encapsulation class of DPCNN network infer.

    Args:
        network (nn.Cell): DPCNN model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids
    """

    def __init__(self, network):
        super(DPCNNInferCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, src_tokens):
        """construct dpcnn infer cell"""
        prediction = self.network(src_tokens)
        predicted_idx = self.log_softmax(prediction)
        predicted_idx, _ = self.argmax(predicted_idx)
        return predicted_idx
