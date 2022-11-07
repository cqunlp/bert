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
"""Negative log likelihood loss between logits and labels."""

from typing import Tuple
from mindspore.common.tensor import Tensor
import mindspore.ops as ops
from mindspore.nn.loss.loss import _Loss

from mindtext.common.utils.class_factory import ClassFactory, ModuleType

@ClassFactory.register(ModuleType.LOSS)
class NegativeLogLikelihood(_Loss):
    r"""
    Gets the negative log likelihood loss between logits and labels.

    Parameters:
        reduction (string) - Apply specific reduction method to the output: 'none', 'mean', 'sum'.
        Default: "mean".

    Inputs:
        input (Tensor) - Input logits, with shape (N,C). Data type only support float32 or float16.
        target (Tensor) - Ground truth labels, with shape (N). Data type only support int32.
        weight (Tensor) - The rescaling weight to each class, with shape (C) and data type only
        support float32 or float16`.

    Outputs:
        Tuple of 2 tensors composed with loss and total_weight.
            loss (Tensor) - when reduction is none and input is 2D tensor, the loss shape is (N,).
                Otherwise, the loss is a scalar. The data type is same with input's.
            total_weight (Tensor) - the total_weight is a scalar. The data type is same with
                weight's.
    """
    def __init__(self, weight: Tensor, reduction: str = 'mean') -> _Loss:
        super(NegativeLogLikelihood, self).__init__(reduction)
        self.weight = weight
        self.nll = ops.NLLLoss(reduction=reduction)

    def construct(self, logits: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """Construct the negative log likelihood loss between logits and labels.

        Inputs:
            - **logits** (Tensor) - The input Tensor. The data type must be float16 or float32.
            - **labels** (Tensor) - The label Tensor which has same shape and data type as `logits`.

        Outputs:
            Tuple of 2 tensors composed with loss and total_weight.
                loss (Tensor) - when reduction is none and input is 2D tensor, the loss shape
                    is (N,). Otherwise, the loss is a scalar. The data type is same with input's.
                total_weight (Tensor) - the total_weight is a scalar. The data type is same with
                    weight's.
        """
        loss, weight = self.nll(logits, labels, self.weight)
        return loss, weight
