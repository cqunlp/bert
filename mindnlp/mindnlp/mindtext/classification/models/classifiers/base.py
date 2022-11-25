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
Base model segmentors
"""
import mindspore.nn as nn


class BaseClassifier(nn.Cell):
    """
    Baseclassifier
    """

    def __init__(self, backbone, neck):
        super(BaseClassifier, self).__init__()
        self.backbone = backbone
        self.neck = neck
        if neck is not None:
            self.with_neck = True
        else:
            self.with_neck = False

    def construct(self, *x):
        x = self.backbone(*x)
        if self.with_neck:
            x = self.neck(x)
        return x
