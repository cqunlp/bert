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
FastText model
"""
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import GradOperation
from mindspore import ParameterTuple
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

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


class FastText(nn.Cell):
    """
    FastText model
    Args:

        vocab_size: vocabulary size
        embedding_dims: The size of each embedding vector
        num_class: number of labels
    """

    def __init__(self, vocab_size, embedding_dims, num_class):
        super(FastText, self).__init__()
        self.vocab_size = vocab_size
        self.embeding_dims = embedding_dims
        self.num_class = num_class
        self.embeding_func = nn.Embedding(vocab_size=self.vocab_size,
                                          embedding_size=self.embeding_dims,
                                          padding_idx=0, embedding_table='Zeros')
        self.fc = nn.Dense(self.embeding_dims, out_channels=self.num_class,
                           weight_init=XavierUniform(1)).to_float(mstype.float16)
        self.reducesum = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=1)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.realdiv = P.RealDiv()
        self.fill = P.Fill()
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, src_tokens, src_token_length):
        """
        construct network
        Args:

            src_tokens: source sentences
            src_token_length: source sentences length

        Returns:
            Tuple[Tensor], network outputs
        """
        src_tokens = self.embeding_func(src_tokens)
        embeding = self.reducesum(src_tokens, 1)

        embeding = self.realdiv(embeding, src_token_length)

        embeding = self.cast(embeding, mstype.float16)
        classifier = self.fc(embeding)
        classifier = self.cast(classifier, mstype.float32)

        return classifier


class FastTextNetWithLoss(nn.Cell):
    """
    Provide FastText training loss

    Args:
        net(nn.Cell): FASTTEXT model.
        loss(Loss): FASTTEXT loss.
    """

    def __init__(self, net, loss):
        super(FastTextNetWithLoss, self).__init__()
        self.fasttext = net
        self.loss_func = loss
        self.squeeze = P.Squeeze(axis=1)
        self.print = P.Print()

    def construct(self, src_tokens, src_tokens_lengths, label_idx):
        """
        FastText network with loss.
        """
        predict_score = self.fasttext(src_tokens, src_tokens_lengths)
        label_idx = self.squeeze(label_idx)
        predict_score = self.loss_func(predict_score, label_idx)

        return predict_score


class FastTextTrainOneStep(nn.Cell):
    """
    Encapsulation class of fasttext network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, net, loss, optimizer, sens=1.0):
        super(FastTextTrainOneStep, self).__init__(auto_prefix=False)
        self.network = FastTextNetWithLoss(net, loss)
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

    def construct(self, src_token_text,
                  src_tokens_text_length,
                  label_idx_tag):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(src_token_text,
                            src_tokens_text_length,
                            label_idx_tag)
        grads = self.grad(self.network, weights)(src_token_text,
                                                 src_tokens_text_length,
                                                 label_idx_tag,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


class FastTextInferCell(nn.Cell):
    """
    Encapsulation class of FastText network infer.

    Args:
        network (nn.Cell): FastText model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids
    """

    def __init__(self, network):
        super(FastTextInferCell, self).__init__(auto_prefix=False)
        self.network = network
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, src_tokens, src_tokens_length):
        """construct fasttext infer cell"""
        prediction = self.network(src_tokens, src_tokens_length)
        predicted_idx = self.log_softmax(prediction)
        predicted_idx, _ = self.argmax(predicted_idx)
        return predicted_idx
