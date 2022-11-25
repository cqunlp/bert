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
"""Updates gradients by the Adaptive Moment Estimation (Adam) algorithm."""

from typing import Union, List
from collections.abc import Iterable
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P, functional as F, composite as C
from mindspore.nn.optim.optimizer import Optimizer
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common.tensor import Tensor
from mindtext.common.utils.class_factory import ClassFactory, ModuleType

_adam_opt = C.MultitypeFuncGraph("adam_opt")

@ClassFactory.register(ModuleType.OPTIMIZER)
class Adam(Optimizer):
    r"""
    Updates gradients by the Adaptive Moment Estimation (Adam) algorithm.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w_{t+1} = w_{t} - l * \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`l` represents scaling factor, :math:`\beta_1, \beta_2` represent
    `beta1` and `beta2`, :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        When separating parameter groups, if you want to centralize the gradient, set grad_centralization to True,
        but the gradient centralization can only be applied to the parameters of the convolution layer.
        If the parameters of the non convolution layer are set to True, an error will be reported.

        To improve parameter groups performance, the customized order of parameters is supported.

        The sparse strategy is applied while the SparseGatherV2 operator is used for forward network.
        The sparse feature is under continuous development. If the sparse strategy wants to be executed on the host,
        set the target to the CPU.

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

            - grad_centralization: Optional. The data type of "grad_centralization" is Bool. If "grad_centralization"
              is in the keys, the set value will be used. If not, the `grad_centralization` is False by default.
              This parameter only works on the convolution layer.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use the dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
                       Default: 0.999.
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0. Default:
                     1e-8.
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking` or `use_nesterov` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Adam(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """
    def __init__(self, params: Union[List[Parameter], List[dict]],
                 learning_rate: Union[float, Tensor, Iterable] = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, use_locking: bool = False,
                 use_nesterov: bool = False, weight_decay: float = 0.0, loss_scale: float = 1.0) -> Optimizer:
        super(Adam, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.cls_name)
        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")
        self.eps = Tensor(eps, mstype.float32)
        self.use_nesterov = use_nesterov
        self.use_locking = use_locking
        self.moment1 = self.parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self.parameters.clone(prefix="moment2", init='zeros')

        self._is_device = True
        self.opt = P.Adam(use_locking, use_nesterov)
        self.sparse_opt = P.FusedSparseAdam(use_locking, use_nesterov)
        self.sparse_opt.add_prim_attr("primitive_target", "CPU")
        self._ps_pull = P.Pull()
        self._ps_push = P.Push("Adam", [0, 1, 2])
        self._ps_push.add_prim_attr("use_nesterov", use_nesterov)

    def construct(self, gradients):
        """Construct the adam optimizer."""
        params = self.parameters
        moment1 = self.moment1
        moment2 = self.moment2
        gradients = self.decay_weight(gradients)
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        gradients = self._grad_sparse_indices_deduplicate(gradients)
        lr = self.get_lr()

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power
        if self.is_group_lr:
            success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push, self._ps_pull,
                                          self.use_locking, self.use_nesterov, self._is_device,
                                          beta1_power, beta2_power, self.beta1, self.beta2, self.eps),
                                lr, gradients, params, moment1, moment2, self.ps_parameters, self.cache_enable)
        else:
            success = self.map_(F.partial(_adam_opt, self.opt, self.sparse_opt, self._ps_push, self._ps_pull,
                                          self.use_locking, self.use_nesterov, self._is_device,
                                          beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr),
                                gradients, params, moment1, moment2, self.ps_parameters, self.cache_enable)
        return success

    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation."""
        self._set_base_target(value)

def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)
