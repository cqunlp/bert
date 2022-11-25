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
"""Bert for Sequence Classification script."""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from mindspore.context import ParallelMode
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import Squeeze
from mindspore.communication.management import get_group_size
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.common.seed import _get_graph_seed
from mindtext.embeddings.bert_embedding import BertEmbedding


class BertforSequenceClassification(nn.Cell):
    """
    Train interface for classification finetuning task.

    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        num_labels (int): Number of label types.
        dropout_prob (float): The dropout probability for BertforSequenceClassification.
        multi_sample_dropout (int): Dropout times per step
    """
    def __init__(self, config, is_training, num_labels, dropout_prob=0.1, multi_sample_dropout=1):
        super(BertforSequenceClassification, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertEmbedding(config, is_training)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.softmax = nn.Softmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(mstype.float32)
        self.dropout_list = []
        for _ in range(0, multi_sample_dropout):
            seed0, seed1 = _get_graph_seed(1, "dropout")
            self.dropout_list.append(ops.Dropout(1-dropout_prob, seed0, seed1))
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.squeeze = Squeeze(1)
        self.num_labels = num_labels
        self.is_training = is_training

    def from_pretrain(self, ckpt_file):
        """
        Load the model parameters from checkpoint
        """
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self, param_dict)

    def init_embedding(self, embedding):
        """
        Manual initialization Embedding
        """
        self.bert = embedding

    def construct(self, input_ids, input_mask, token_type_id, label_ids=0):
        """
        Classification task
        """
        _, pooled_output = self.bert(input_ids, token_type_id, input_mask)
        loss = None
        if self.is_training:
            for dropout in self.dropout_list:
                cls, _ = dropout(pooled_output)
                logits = self.dense_1(cls)
                temp_loss = self.loss(logits, self.squeeze(label_ids))
                if loss is None:
                    loss = temp_loss
                else:
                    loss += temp_loss
            loss = loss/len(self.dropout_list)
        else:
            loss = self.dense_1(pooled_output)
        return loss


class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


class BertFinetuneCell(nn.Cell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):

        super(BertFinetuneCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """
        Bert Finetune

        Returns:
            loss: loss from network output
        """
        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        if not self.gpu_target:
            init = self.alloc_status()
            init = F.depend(init, loss)
            clear_status = self.clear_status(init)
            scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))

        self.optimizer(grads)
        return loss
