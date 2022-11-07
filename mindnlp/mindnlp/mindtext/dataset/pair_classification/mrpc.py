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
    MRPC dataset
"""
import os
from typing import Union, Dict, Optional

import pandas as pd
from pandas import DataFrame

from tqdm import tqdm
import mindspore.dataset as ds

from ..base_dataset import PairCLSBaseDataset
from ..utils import get_split_func
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class MRPCDataset(PairCLSBaseDataset):
    """
    MRPC dataset.

    Args:
        paths (str): Dataset file path or Dataset directory path.
        tokenizer (Union[str]): Tokenizer function, default 'spacy'.
        lang (str): Tokenizer language, default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.
        train_ratio (float): The ratio of the train set and the ratio of dev set is 1-train_ratio,default 0.8.

    Examples:
        >>> mrpc = MRPCDataset(tokenizer='spacy', lang='en', train_ratio=0.8, paths=Path to the dataset folder)
          # mrpc = MRPCDataset(tokenizer='spacy', lang='en', buckets=[16,32,64])
        >>> dataset = mrpc()
    """

    def __init__(self, paths: str, tokenizer: Union[str] = 'spacy', lang: str = 'en', max_size: Optional[int] = None,
                 min_freq: Optional[int] = None, padding: str = '<pad>', unknown: str = '<unk>',
                 train_ratio: float = 0.8, **kwargs):
        super(MRPCDataset, self).__init__(sep='\t', name='MRPC', **kwargs)
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        self._train_ratio = train_ratio

    def __call__(self) -> Dict[str, ds.MindDataset]:
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self._mind_datasets

    def _load(self, path: str) -> DataFrame:
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            dataset = pd.read_csv(f, sep='\n', names=['label', 'ID1', 'ID2', 'sentence1', 'sentence2'])
            columns = dataset.columns.values
            tqdm.pandas(desc=f"{self._name} dataset loadding")
            split_row = get_split_func(dataset, '\t')
            dataset = dataset.progress_apply(split_row, axis=1, result_type="expand")
            dataset.columns = columns
            dataset.fillna('')
            dataset.dropna(inplace=True)
        return dataset

    def load(self, paths: Optional[str] = None) -> Dict[str, DataFrame]:
        """
        Load MRPC dataset.

        Args:
            paths (str, Optional): MRPC dataset directory path, default None.

        Returns:
            Dict[str, DataFrame]:  A MRPC dataset dict.
        """
        if paths is None:
            self.download()
        if not os.path.isdir(paths):
            raise NotADirectoryError(f"{paths} is not a valid directory.")

        files = {'train': "msr_paraphrase_train.txt",
                 "test": "msr_paraphrase_test.txt"}

        dev_ids_file = "mrpc_dev_ids.tsv"

        self._datasets = {}
        for name, filename in files.items():
            filepath = os.path.join(paths, filename)
            if not os.path.isfile(filepath):
                if 'test' not in name:
                    raise FileNotFoundError(f"{name} not found in directory {filepath}.")
            dataset = self._load(filepath)
            if 'train' in name:
                dev_ids_file_path = os.path.join(paths, dev_ids_file)
                dev_ids = pd.read_csv(dev_ids_file_path, sep='\n')
                dev_ids = dev_ids.values.tolist()
                dev_dataset = pd.DataFrame(columns=['label', 'sentence1', 'sentence2'])
                train_dataset = pd.DataFrame(columns=['label', 'sentence1', 'sentence2'])
                for row in dataset.itertuples():
                    ids = [row.ID1 + '\t' + row.ID2]
                    if ids in dev_ids:
                        dev_dataset = dev_dataset.append(pd.DataFrame(
                            {'label': [row.label],
                             'sentence1': [row.sentence1],
                             'sentence2': [row.sentence2]}))
                    else:
                        train_dataset = train_dataset.append(pd.DataFrame(
                            {'label': [row.label],
                             'sentence1': [row.sentence1],
                             'sentence2': [row.sentence2]}))

                self._datasets['train'] = train_dataset
                self._datasets['dev'] = dev_dataset
            else:
                self._datasets[name] = dataset.reset_index(drop=True)
        return self._datasets

    def download(self):
        """
        Cannot download MRPC automatically.
        """
        raise RuntimeError("MRPC cannot be downloaded automatically.")
