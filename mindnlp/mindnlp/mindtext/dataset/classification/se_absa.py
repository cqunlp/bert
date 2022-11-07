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
    SE-ABSA16_PHNS dataset
"""
from typing import Union, Dict, Optional

import pandas as pd
from pandas import DataFrame

import mindspore.dataset as ds

from ..base_dataset import CLSBaseDataset
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class SEABSA16Dataset(CLSBaseDataset):
    """
    SE-ABSA16_PHNS dataset load.

    Args:
        paths (Union[str, Dict[str, str]], Optional): Dataset file path or Dataset directory path, default None.
        tokenizer (Union[str]): Tokenizer function,default 'spacy'.
        lang (str): Tokenizer language,default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token,default `<pad>`.
        unknown (str): Unknown token,default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.
    Examples:
        >>> SEABSA16Dataset = SEABSA16Dataset(tokenizer='spacy', lang='en',paths=Path to the dataset folder)
        >>> dataset = SEABSA16Dataset()
    """

    def __init__(self, paths: Optional[Union[str, Dict[str, str]]] = None,
                 tokenizer: Union[str] = 'spacy', lang: str = 'en', max_size: Optional[int] = None,
                 min_freq: Optional[int] = None,
                 padding: str = '<pad>', unknown: str = '<unk>', **kwargs):
        super(SEABSA16Dataset, self).__init__(sep='\t', name='SE-ABSA16_PHNS', **kwargs)
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown

    def __call__(self) -> Dict[str, ds.MindDataset]:
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self._mind_datasets

    def _load(self, path: str) -> DataFrame:
        """
        Load dataset from SE_ABSA16 file.

        Args:
            path (str): Dataset file path.

        Returns:
            DataFrame: Dataset file will be read as a DataFrame.
        """
        with open(path, 'r', encoding='utf-8') as f:
            cls = len(f.readline().strip().split())
        with open(path, 'r', encoding='utf-8') as f:
            columns = f.readline().strip().split('\t')
            if "label" in columns:
                if cls == 2:
                    dataset = pd.read_csv(f, sep='\t', names=['label', 'sentence'])
                else:
                    dataset = pd.read_csv(f, sep='\t', names=['index', 'label', 'sentence'])
                dataset = dataset[['sentence', 'label']]
            else:
                dataset = pd.read_csv(f, sep='\t', names=['index', 'sentence'])
        dataset.fillna('')
        dataset.dropna(inplace=True)
        return dataset
