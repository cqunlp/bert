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
    luke for tagging and reading comprehension tasks
"""
from typing import Optional

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore import context
from mindtext.modules.encoder.luke import LukeEntityAwareAttentionModel

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
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


class LukeForReadingComprehension(nn.Cell):
    """Luke for reading comprehension task"""

    def __init__(self, config):
        super(LukeForReadingComprehension, self).__init__()
        self.luke = LukeEntityAwareAttentionModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, 2)
        self.split = ops.Split(-1, 2)
        self.squeeze = ops.Squeeze(-1)
        self.shape = ops.Shape()

    def construct(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask
    ):
        """LukeForReadingComprehension construct"""
        encoder_outputs = self.luke(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        word_hidden_states = encoder_outputs[0][:, : ops.shape(word_ids)[1], :]
        logits = self.qa_outputs(word_hidden_states)
        start_logits, end_logits = self.split(logits)
        start_logits = self.squeeze(start_logits)
        end_logits = self.squeeze(end_logits)
        return start_logits, end_logits


class LukeForReadingComprehensionWithLoss(nn.Cell):
    """LukeForReadingComprehensionWithLoss"""

    def __init__(self, net, loss: Optional[nn.Cell] = None):
        super(LukeForReadingComprehensionWithLoss, self).__init__()
        self.net = net
        if isinstance(loss, nn.Cell):
            self.loss = loss
        else:
            self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        self.squeeze = ops.Squeeze(-1)
        self.shape = ops.Shape()
        self.clamp = ops.clip_by_value

    def construct(
            self,
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
            start_positions=None,
            end_positions=None
    ):
        """
        LukeForReadingComprehensionWithLoss's construct
        """
        start_logits, end_logits = self.net(word_ids,
                                            word_segment_ids,
                                            word_attention_mask,
                                            entity_ids,
                                            entity_position_ids,
                                            entity_segment_ids,
                                            entity_attention_mask)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.shape) > 1:
                start_positions = self.squeeze(start_positions)
            if len(end_positions.shape) > 1:
                end_positions = self.squeeze(end_positions)

            ignored_index = self.shape(start_logits)[1]

            start_positions = self.clamp(start_positions, 0, ignored_index)
            end_positions = self.clamp(end_positions, 0, ignored_index)
            start_loss = self.loss(start_logits, start_positions)
            end_loss = self.loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0
            outputs = total_loss

        else:
            outputs = (start_logits, end_logits,)

        return outputs


class LukeSquadCell(nn.Cell):
    """
    LukeSquadCell for Train
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(LukeSquadCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
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
                  word_ids,
                  word_segment_ids,
                  word_attention_mask,
                  entity_ids,
                  entity_position_ids,
                  entity_segment_ids,
                  entity_attention_mask,
                  start_positions,
                  end_positions,
                  sens: Optional[int] = None
                  ):
        """LukeSquad"""
        weights = self.weights
        init = False
        loss = self.network(word_ids,
                            word_segment_ids,
                            word_attention_mask,
                            entity_ids,
                            entity_position_ids,
                            entity_segment_ids,
                            entity_attention_mask,
                            start_positions,
                            end_positions)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = ops.tuple_to_array((sens,))
        if not self.gpu_target:
            init = self.alloc_status()
            init = F.depend(init, loss)
            clear_status = self.clear_status(init)
            scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(word_ids,
                                                 word_segment_ids,
                                                 word_attention_mask,
                                                 entity_ids,
                                                 entity_position_ids,
                                                 entity_segment_ids,
                                                 entity_attention_mask,
                                                 start_positions,
                                                 end_positions, self.cast(scaling_sens,
                                                                          mstype.float32))
        self.optimizer(grads)
        cond = False
        return (loss, cond)
