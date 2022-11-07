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
"""Implements the Adam algorithm to fix the weight decay."""

from typing import Union, List
from collections.abc import Iterable
import numpy as np
from mindspore.ops import functional as F, composite as C
from mindspore.nn.optim.optimizer import Optimizer
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindtext.common.utils.class_factory import ClassFactory, ModuleType

_adam_opt = C.MultitypeFuncGraph("adam_opt")

@ClassFactory.register(ModuleType.OPTIMIZER)
class AdamWeightDecay(Optimizer):
    r"""
    Implements the Adam algorithm to fix the weight decay.

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            update = \frac{m_{t+1}}{\sqrt{v_{t+1}} + eps} \\
            update =
            \begin{cases}
                update + weight\_decay * w_{t}
                    & \text{ if } weight\_decay > 0 \\
                update
                    & \text{ otherwise }
            \end{cases} \\
            w_{t+1}  = w_{t} - lr * update
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`lr` represents `learning_rate`,
    :math:`\beta_1, \beta_2` represent `beta1` and `beta2`, :math:`t` represents updating step while
    :math:`w` represents `params`.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" is in the keys, the value of the corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" is in the keys, the value of the corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" is in the keys, the value must be the order of parameters and
              the order will be followed in the optimizer. There are no other keys in the `dict` and the parameters
              which in the 'order_params' must be in one of group parameters.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use the dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.AdamWeightDecay(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.AdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
   """
    def __init__(self, params: Union[List[Parameter], List[dict]],
                 learning_rate: Union[float, Tensor, Iterable] = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-6, weight_decay: float = 0.0) -> Optimizer:
        super(AdamWeightDecay, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')

    def construct(self, gradients):
        """Construct the adamW optimizer."""
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map_reverse(F.partial(_adam_opt, self.beta1, self.beta2, self.eps),
                                                      lr, self.weight_decay, self.parameters, self.moments1,
                                                      self.moments2, gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map_reverse(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                                      self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                      gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map_reverse(F.partial(_adam_opt, self.beta1, self.beta2, self.eps, lr,
                                                            self.weight_decay),
                                                  self.parameters, self.moments1, self.moments2,
                                                  gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result

def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)
