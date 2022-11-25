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
# ==============================================================================
"""Softmax cross entropy between logits and labels."""

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.nn.loss.loss import _Loss
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._checkparam import Validator as validator
from mindspore import context
from mindtext.common.utils.class_factory import ClassFactory, ModuleType

@ClassFactory.register(ModuleType.LOSS)
class CrossEntropy(_Loss):
    r"""
    Computes softmax cross entropy between logits and labels.

    Measures the distribution error between the probabilities of the input (computed with softmax
    function) and the target where the classes are mutually exclusive (only one class is positive)
    using cross entropy loss.

    Typical input into this function is unnormalized scores denoted as x whose shape is (N, C),
    and the corresponding targets.

    For each instance :math:`x_i`, i ranges from 0 to N-1, the loss is given as:

    .. math::
        \ell(x_i, c) = - \log\left(\frac{\exp(x_i[c])}{\sum_j \exp(x_i[j])}\right)
        =  -x_i[c] + \log\left(\sum_j \exp(x_i[j])\right)

    where :math:`x_i` is a 1D score Tensor, :math:`c` is the index of 1 in one-hot.

    Note:
        While the target classes are mutually exclusive, i.e., only one class is positive in the
        target, the predicted probabilities need not to be exclusive. It is only required that
        the predicted probability distribution of entry is a valid one.

    Args:
        sparse (bool): Specifies whether labels use sparse format or not. Default: False.
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean",
            "sum", and "none". If "none", do not perform reduction. Default: "none".

    Inputs:
        logits (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
        labels (Tensor) - Tensor of shape (N, ). If sparse is True, The type of labels is int32 or
        int64.
        Otherwise, the type of labels is the same as the type of logits.

    Outputs:
        Tensor, a tensor of the same shape and type as logits with the component-wise logistic
        losses.

    Raises:
        TypeError: If `sparse` is not a bool.
        TypeError: If `sparse` is True and dtype of `labels` is neither int32 not int64.
        TypeError: If `sparse` is False and dtype of `labels` is neither float16 not float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels)
        >>> print(output)
        [67.]
    """
    def __init__(self,
                 sparse: bool = False,
                 reduction: str = 'none') -> _Loss:
        super(CrossEntropy, self).__init__(reduction)
        self.sparse = validator.check_bool(sparse, "sparse")
        self.reduction = reduction
        self.softmax_cross_entropy = nn.SoftmaxCrossEntropyWithLogits()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0., mstype.float32)
        self.is_cpugpu = context.get_context('device_target') in ["CPU", "GPU"]
        self.sparse_softmax_cross_entropy = P.SparseSoftmaxCrossEntropyWithLogits()

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Construct the softmax cross entropy between logits and labels.

        Inputs:
            - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
            - **labels** (Tensor) - Tensor of shape (N, ). If `sparse` is True, The type of
            `labels` is int32 or int64. If `sparse` is False, the type of `labels` is the same as
            the type of `logits`.

        Outputs:
            Tensor, a tensor of the same shape and type as logits with the component-wise logistic
            losses.
        """
        if self.sparse:
            if self.reduction == 'mean':
                x = self.sparse_softmax_cross_entropy(logits, labels)
                return x
            labels = self.one_hot(labels, F.shape(logits)[-1], self.on_value, self.off_value)
        x = self.softmax_cross_entropy(logits, labels)[0]
        return self.get_loss(x)
