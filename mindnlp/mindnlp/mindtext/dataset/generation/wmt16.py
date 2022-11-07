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
    WMT16 dataset(contains ro-en and de-en)
"""
import os.path
from typing import Union, Optional, Dict

import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
import mindspore.dataset as ds

from ..base_dataset import GenerateBaseDataset
from ..utils import check_loader_paths
from ..utils import _preprocess_sequentially, get_tokenizer, _get_dataset_type
from .. import Vocabulary, Pad
from .. import ClassFactory, ModuleType

LANGUAGE_PAIRS = ['ro-en', 'de-en']
LANGUAGE_PAIRS_TRANS = []
for i in LANGUAGE_PAIRS:
    a, b = i.split('-')
    LANGUAGE_PAIRS_TRANS.append(b + '-' + a)


@ClassFactory.register(ModuleType.DATASET)
class WMT16Dataset(GenerateBaseDataset):
    """
    WMT16 dataset.

    Args:
        path (str, Optional): Dataset file path or Dataset directory path, default None.
        tokenizer (Union[str]): Tokenizer function, default 'spacy'.
        lang (str): Tokenizer language, default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        language_pairs (str): The language pairs of wmt16. default `ro-en`.

    Examples:
        >>> wmt16 = WMT16Dataset(tokenizer='facebook/bart', language_pairs='ro-en')
          # wmt16 = WMT16Dataset(tokenizer='facebook/bart', language_pairs='ro-en')
          # wmt16 = WMT16Dataset(language_pairs='ro-en')
        >>> dataset = wmt16()
    """

    def __init__(self, paths: Optional[str] = None, tokenizer: Union[str] = 'spacy', lang: str = 'en',
                 max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>', language_pairs: str = 'ro-en', **kwargs):
        super(WMT16Dataset, self).__init__(name='WMT16', **kwargs)
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        if language_pairs not in LANGUAGE_PAIRS + LANGUAGE_PAIRS_TRANS:
            raise AttributeError(f"`language_pairs` should be in `${LANGUAGE_PAIRS + LANGUAGE_PAIRS_TRANS}`")
        self._source, self._target = language_pairs.split("-")
        self._stream = language_pairs in ['en-de', 'de-en']
        self._process4transformer = language_pairs in ['en-de', 'de-en']
        self._tokenize = 'raw' if language_pairs in ['en-de', 'de-en'] else tokenizer

    def __call__(self):
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self._mind_datasets

    def process(self, tokenizer: Union[str], lang: str, max_size: Optional[int] = None, min_freq: Optional[int] = None,
                padding: str = "<pad>", unknown: str = "<unk>", **kwargs) -> Dict[str, ds.MindDataset]:
        """
        Preprocess dataset.

        Args:
            tokenizer (Union[str]): Tokenizer function.
            lang (str): Tokenizer language.
            max_size (int, Optional): Vocab max size, default None.
            min_freq (int, Optional): Min word frequency, default None.
            padding (str): Padding token,default `<pad>`.
            unknown (str): Unknown token,default `<unk>`.

        Returns:
            Dict[str, MindDataset]: A MindDataset dictionary.
        """
        self._buckets = kwargs.pop("buckets", self._buckets)
        if isinstance(tokenizer, str):
            self._tokenizer = get_tokenizer(tokenizer, lang=lang)
        else:
            self._tokenizer = None

        dataset_file_name = _preprocess_sequentially(list(self._datasets.keys()))

        for dataset_name in dataset_file_name:
            dataset = self._datasets.get(dataset_name)
            d_t = _get_dataset_type(dataset_name)
            if isinstance(dataset, DataFrame):
                if self._stream:
                    if self._process4transformer:
                        preprocess_func = self._transfomer_process(dataset, dataset_type=d_t)
                    else:
                        preprocess_func = self._stream_process(dataset, max_size, min_freq, padding, unknown,
                                                               dataset_type=d_t)
                    dataset = self.convert_to_mr(dataset, dataset_name, is_test=d_t == "test",
                                                 process_function=preprocess_func)
                else:
                    dataset = self._process(dataset, max_size, min_freq, padding, unknown, dataset_type=d_t)
                    dataset = self.convert_to_mr(dataset, dataset_name, is_test=d_t == "test")
                self._mind_datasets[dataset_name] = dataset
        del self._datasets
        return self._mind_datasets

    def _transfomer_process(self, dataset: DataFrame, dataset_type: str) -> callable:
        """
        Preprocess dataset for transformer model.
        """
        if isinstance(self._tokenizer, PreTrainedTokenizerBase):
            raise TypeError(
                "When `process4transformer` is `True`, `tokenizer` can not be assigned a pretrained tokenizer")
        dataset['document'] = self.tokenize_progress(dataset, dataset_type, 'document')
        dataset['summary'] = self.tokenize_progress(dataset, dataset_type, 'summary')
        vocab = Vocabulary.from_file(self._vocab_file)
        eos = '</s>'
        sos = '<s>'

        if not self._buckets:
            pad1 = Pad(max_length=self._max_length, pad_val=0, truncate=self._truncation_strategy)
            pad2 = Pad(max_length=self._max_pair_length, pad_val=0, truncate=self._truncation_strategy)

            def token_to_idx(row):
                source_sos_tokens = [sos] + row['document']
                source_eos_tokens = row['document'] + [eos]
                source_length = len(source_eos_tokens)
                target_sos_tokens = [sos] + row['summary']
                target_eos_tokens = row['summary'] + [eos]
                target_length = len(target_eos_tokens)
                source_mask = pad1([1 for _ in range(source_length)])
                target_mask = pad2([1 for _ in range(target_length)])
                data = {'source_sos_ids': pad1([vocab[token] for token in source_sos_tokens]),
                        'source_sos_mask': source_mask,
                        'source_eos_ids': pad1([vocab[token] for token in source_eos_tokens]),
                        'source_eos_mask': source_mask,
                        'target_sos_ids': pad2([vocab[token] for token in target_sos_tokens]),
                        'target_sos_mask': target_mask,
                        'target_eos_ids': pad2([vocab[token] for token in target_eos_tokens]),
                        'target_eos_mask': target_mask}
                return data
        else:
            pad = Pad(pad_val=0, buckets=self._buckets, truncate=self._truncation_strategy)

            def token_to_idx(row):
                source_sos_tokens = [sos] + row['document']
                source_eos_tokens = row['document'] + [eos]
                source_length = len(source_eos_tokens)
                target_sos_tokens = [sos] + row['summary']
                target_eos_tokens = row['summary'] + [eos]
                target_length = len(target_eos_tokens)
                source_mask = pad([1 for _ in range(source_length)])
                target_mask = pad([1 for _ in range(target_length)])
                data = {'source_sos_ids': pad([vocab[token] for token in source_sos_tokens]),
                        'source_sos_mask': source_mask,
                        'source_eos_ids': pad([vocab[token] for token in source_eos_tokens]),
                        'source_eos_mask': source_mask,
                        'target_sos_ids': pad([vocab[token] for token in target_sos_tokens]),
                        'target_sos_mask': target_mask,
                        'target_eos_ids': pad([vocab[token] for token in target_eos_tokens]),
                        'target_eos_mask': target_mask}
                if len(data["source_sos_ids"]) > len(data['target_sos_ids']):
                    data['target_sos_ids'] = Pad.padding(data['target_sos_ids'], len(data['source_sos_ids']), 0)
                    data['target_eos_ids'] = data['target_sos_ids']
                    data['target_sos_mask'] = Pad.padding(target_mask, len(data['source_sos_ids']), 0)
                    data['target_eos_mask'] = data['target_sos_mask']
                else:
                    data['source_sos_ids'] = Pad.padding(data['source_sos_ids'], len(data['target_sos_ids']), 0)
                    data['source_eos_ids'] = data['source_sos_ids']
                    data['source_sos_mask'] = Pad.padding(source_mask, len(data['target_sos_ids']), 0)
                    data['source_eos_mask'] = data['source_sos_mask']
                data['padding_length'] = len(data['source_sos_ids'])
                return data
        return token_to_idx

    def _load(self, path: str) -> DataFrame:
        _, filename = os.path.split(path)
        _, suffix = os.path.splitext(filename)
        if suffix == '.tsv':
            dataset = pd.read_csv(path, sep='\t', keep_default_na=False)[:1000]
        else:
            dataset = pd.read_csv(path, keep_default_na=False)
        column_names = {self._source: 'document', self._target: 'summary'}
        dataset_names = dataset.columns
        dataset.columns = [column_names[dataset_names[0]], column_names[dataset_names[1]]]
        return dataset

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> Dict[str, DataFrame]:
        if not paths:
            paths = self.download()
        language_pair = self._source + '-' + self._target
        language_pair_trans = self._target + '-' + self._source
        language_pair = language_pair if language_pair in LANGUAGE_PAIRS else language_pair_trans
        dataset_directory_path = os.path.join(paths, language_pair)

        if not self._vocab_file:
            file_list = os.listdir(dataset_directory_path)
            file_name = None
            for name in file_list:
                if "vocab" in name:
                    file_name = name
                    break
            if isinstance(file_name, str):
                self._vocab_file = os.path.join(dataset_directory_path, file_name)

        if self._process4transformer and not isinstance(self._vocab_file, str):
            raise TypeError("When `process4transformer` is `True`, need to give a vocab file path")

        paths = check_loader_paths(dataset_directory_path)
        self._datasets = {name: self._load(path) for name, path in paths.items()}
        return self._datasets
