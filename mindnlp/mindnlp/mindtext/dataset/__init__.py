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
    Dataset init
"""
from os.path import dirname, join, split, splitext
from glob import glob
from keyword import iskeyword

from ..common import Vocabulary, Pad
from ..common.utils.class_factory import ClassFactory, ModuleType

DATASET_CLS = ['classification', 'pair_classification', 'regression', 'tagging', 'reading_comprehension', 'generation']

for dataset_class in DATASET_CLS:
    basedir = join(dirname(__file__), dataset_class)

    for name in glob(join(basedir, '*.py')):
        module = splitext(split(name)[-1])[0]
        if not module.startswith('_') and \
                module.isidentifier() and \
                not iskeyword(module):
            __import__(__name__ + '.' + dataset_class + '.' + module)
