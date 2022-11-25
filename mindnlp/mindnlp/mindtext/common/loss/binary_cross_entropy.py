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
"""Binary cross entropy between the true labels and predicted labels."""

from mindspore.common.tensor import Tensor
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P

from mindtext.common.utils.class_factory import ClassFactory, ModuleType

@ClassFactory.register(ModuleType.LOSS)
class BinaryCrossEntropy(_Loss):
    r"""
    BCELoss creates a criterion to measure the binary cross entropy between the true labels and
    predicted labels.

    Set the predicted labels as :math:`x`, true labels as :math:`y`, the output loss as :math:
    `\ell(x, y)`. Let,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Note:
        Note that the predicted labels should always be the output of sigmoid and the true labels
        should be numbers between 0 and 1.

    Args:
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            And it must have same shape and data type as `inputs`. Default: None
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.

    Inputs:
        - **logits** (Tensor) - The input Tensor. The data type must be float16 or float32.
        - **labels** (Tensor) - The label Tensor which has same shape and data type as `logits`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape
        as `logits`. Otherwise, the output is a scalar.

    Raises:
        TypeError: If dtype of `logits`, `labels` or `weight` (if given) is neither float16 not
            float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If shape of `logits` is not the same as `labels` or `weight` (if given).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> weight = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 3.3, 2.2]]), mindspore.float32)
        >>> loss = nn.BCELoss(weight=weight, reduction='mean')
        >>> logits = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        1.8952923
    """
    def __init__(self, weight: Tensor = None, reduction: str = 'none') -> _Loss:
        super(BinaryCrossEntropy, self).__init__(reduction)
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction=reduction)
        self.weight_one = weight is None
        if not self.weight_one:
            self.weight = weight
        else:
            self.ones = P.OnesLike()

    def construct(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Construct the binary cross entropy between the true labels and predicted logits.

        Inputs:
            - **logits** (Tensor) - The input Tensor. The data type must be float16 or float32.
            - **labels** (Tensor) - The label Tensor which has same shape and data type as `logits`.

        Outputs:
            Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same
            shape as `logits`. Otherwise, the output is a scalar.
        """
        if self.weight_one:
            weight = self.ones(logits)
        else:
            weight = self.weight
        loss = self.binary_cross_entropy(logits, labels, weight)
        return loss
