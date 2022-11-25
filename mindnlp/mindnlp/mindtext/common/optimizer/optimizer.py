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
 Produce the optimizer
"""
import os
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype

from mindtext.classification.utils.lr_schedule import polynomial_decay_scheduler


def create_optimizer(param, config):
    """
    create optimizer
    """
    config_optimizer = config.OPTIMIZER
    if config_optimizer.function == "Adam":
        if "poly_lr_scheduler_power" in config_optimizer.keys():
            return custom_optimizer(param, config)
        optimizer = nn.optim.Adam(param, config_optimizer.lr)
        return optimizer
    return None


def custom_optimizer(param, config):
    """
    custom optimizer
    """
    config_optimizer = config.OPTIMIZER
    update_steps = config.TRAIN.epochs * config.OPTIMIZER.decay_steps
    rank_size = os.getenv("RANK_SIZE")
    if isinstance(rank_size, int):
        raise ValueError("RANK_SIZE must be integer")
    if rank_size is not None and int(rank_size) > 1:
        base_lr = config_optimizer.lr
    else:
        base_lr = config_optimizer.lr / 10
    lr = Tensor(polynomial_decay_scheduler(lr=base_lr,
                                           min_lr=config_optimizer.min_lr,
                                           decay_steps=config_optimizer.decay_steps,
                                           total_update_num=update_steps,
                                           warmup_steps=config_optimizer.warmup_steps,
                                           power=config_optimizer.poly_lr_scheduler_power), dtype=mstype.float32)
    optimizer = nn.optim.Adam(param, lr, beta1=0.9, beta2=0.999)

    return optimizer
