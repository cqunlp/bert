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
"""Builder of optimizer."""

import inspect
import mindspore as ms

from mindtext.common.utils.class_factory import ClassFactory, ModuleType


def build_optimizer(cfg):
    """Builder of box assigner."""
    return ClassFactory.get_instance_from_cfg(cfg, ModuleType.OPTIMIZER)


def register_optimizer():
    """register optimizer"""
    for module_name in dir(ms.nn):
        if module_name.startswith('__'):
            continue
        metric = getattr(ms.nn, module_name)
        if inspect.isclass(metric) and issubclass(metric, ms.nn.Metric):
            ClassFactory.register_cls(metric, ModuleType.OPTIMIZER)
