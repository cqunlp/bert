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
    QQP dataset
"""
from typing import Union, Dict, Optional

from tqdm import tqdm
import pandas as pd
from pandas import DataFrame

from ..base_dataset import PairCLSBaseDataset
from ..utils import get_split_func
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class QQPDataset(PairCLSBaseDataset):
    """
    QQP dataset.

    Args:
        paths (Union[str, Dict[str, str]], Optional): Dataset file path or Dataset directory path, default None.
        tokenizer (Union[str]): Tokenizer function, default 'spacy'.
        lang (str): Tokenizer language, default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.

    Examples:
        >>> qqp = QQPDataset(tokenizer='spacy', lang='en')
          # qqp = QQPDataset(tokenizer='spacy', lang='en', buckets=[16,32,64])
        >>> dataset = qqp()
    """

    def __init__(self, paths: Optional[Union[str, Dict[str, str]]] = None,
                 tokenizer: Union[str] = 'spacy', lang: str = 'en', max_size: Optional[int] = None,
                 min_freq: Optional[int] = None, padding: str = '<pad>', unknown: str = '<unk>', **kwargs):
        super(QQPDataset, self).__init__(sep='\t', name='QQP', **kwargs)
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown

    def __call__(self):
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self._mind_datasets

    def _load(self, path: str) -> DataFrame:
        with open(path, 'r', encoding='utf-8') as f:
            columns = f.readline().strip().split('\t')
            dataset = pd.read_csv(f, sep='\n', names=columns)
            tqdm.pandas(desc=f"{self._name} dataset loadding")
            split_row = get_split_func(dataset, '\t')
            if "is_duplicate" in dataset.columns.values:
                dataset = dataset.progress_apply(split_row, axis=1, result_type="expand")
                dataset.columns = columns
                dataset = dataset[['question1', 'question2', 'is_duplicate']]
                dataset.columns = ['sentence1', 'sentence2', 'label']
            else:
                dataset = dataset.progress_apply(split_row, axis=1, result_type="expand")
                dataset.columns = columns
                dataset = dataset[['question1', 'question2']]
                dataset.columns = ['sentence1', 'sentence2']
            dataset.fillna('')
            dataset.dropna(inplace=True)
        return dataset
