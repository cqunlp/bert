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
    ELI5 dataset
"""
import os
from typing import Union, List, Dict, Optional

from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from .. import Vocabulary, Pad
from ..base_dataset import GenerateBaseDataset
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class ELI5Dataset(GenerateBaseDataset):
    """
    ELI5 dataset.

    Args:
        path (str, Optional): Dataset file path or Dataset directory path, default None.
        tokenizer (Union[str]): Tokenizer function, default 'spacy'.
        lang (str): Tokenizer language, default 'en'.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.

    Examples:
        >>> eli5 = ELI5Dataset(tokenizer='facebook/bart')
          # eli5 = ELI5Dataset(tokenizer='facebook/bart')
        >>> dataset = eli5()
    """

    def __init__(self, paths: Optional[str] = None, tokenizer: Union[str] = 'spacy', lang: str = 'en',
                 max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>', **kwargs):
        super(ELI5Dataset, self).__init__(name='ELI5', stream=True, **kwargs)
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

    def load(self, paths: str = None) -> Dict[str, DataFrame]:
        """
        Load ELI5 dataset.

        Args:
            paths (str, Optional): ELI5 dataset directory path, default None.

        Returns:
            Dict[str, DataFrame]: A ELI5 dataset dict.
        """
        if paths is None:
            paths = self.download()
        if not os.path.isdir(paths):
            raise NotADirectoryError(f"{paths} is not a valid directory.")

        files = {'dev_eli5': 'dev_eli5.csv',
                 'dev_askh': 'dev_askh.csv',
                 'dev_asks': 'dev_asks.csv',
                 'test_eli5': 'test_eli5.csv',
                 'test_askh': 'test_askh.csv',
                 'test_asks': 'test_asks.csv',
                 'train_eli5': 'train_eli5.csv',
                 'train_askh': 'train_askh.csv',
                 'train_asks': 'train_asks.csv'}

        self._datasets = {}
        for name, filename in files.items():
            filepath = os.path.join(paths, filename)
            if not os.path.isfile(filepath):
                if 'test' not in name:
                    raise FileNotFoundError(f"{name} not found in directory {filepath}.")
            self._datasets[name] = self._load(filepath)
        train = []
        dev = []
        test = []
        dataset_names = list(self._datasets.keys())
        for i in dataset_names:
            if 'train' in i:
                train.append(self._datasets[i])
            if 'dev' in i:
                dev.append(self._datasets[i])
            if 'test' in i:
                test.append(self._datasets[i])
            del self._datasets[i]
        self._datasets['train'] = pd.concat(train, axis=0)
        self._datasets['dev'] = pd.concat(dev, axis=0)
        self._datasets['test'] = pd.concat(test, axis=0)
        return self._datasets

    def _load(self, path: str) -> DataFrame:
        dataset = pd.read_csv(path, keep_default_na=False)
        tqdm.pandas(desc=f"{self._name} dataset loadding")

        def split_str2liststr(x):
            # $%$ is separator
            return [i for i in x.split("$%$")]

        def split_str2listint(x):
            # $%$ is separator
            return [int(i) for i in x.split("$%$")]

        a_id = pd.DataFrame(dataset['a_id'].progress_apply(split_str2liststr))
        dataset.drop('a_id', axis=1, inplace=True)
        a_id = a_id.explode('a_id')
        a_id['index_group'] = a_id.index
        a_id = a_id.reset_index(drop=True)
        a_id['index'] = a_id.index

        text = pd.DataFrame(dataset['text'].progress_apply(split_str2liststr))
        dataset.drop('text', axis=1, inplace=True)
        text = text.explode('text')
        text = text.reset_index(drop=True)
        text['index'] = text.index

        score = pd.DataFrame(dataset['score'].progress_apply(split_str2listint))
        dataset.drop('score', axis=1, inplace=True)
        score = score.explode('score')
        score = score.reset_index(drop=True)
        score['index'] = score.index

        dataset['answers_urls'] = dataset['answers_urls'].map(split_str2liststr)
        dataset['selftext_urls'] = dataset['selftext_urls'].map(split_str2liststr)
        dataset['title_urls'] = dataset['title_urls'].map(split_str2liststr)

        def concat_title_selftext(x):
            return x['title'] + ' ' + x['selftext']

        dataset['question'] = dataset[['title', 'selftext']].progress_apply(concat_title_selftext, axis=1,
                                                                            result_type='expand')
        dataset.drop('title', axis=1, inplace=True)
        dataset.drop('selftext', axis=1, inplace=True)

        answers = pd.merge(a_id, text, on='index')
        answers = pd.merge(answers, score, on='index')
        answers.drop('index', axis=1, inplace=True)
        answers.index = answers['index_group']
        answers.drop('index_group', axis=1, inplace=True)

        dataset = dataset.join(answers)
        return dataset

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset['question'] = self.tokenize_progress(dataset, dataset_type, 'question')
            dataset['text'] = self.tokenize_progress(dataset, dataset_type, 'text')

            if dataset_type == 'train':
                self._vocab = Vocabulary.from_dataset(dataset, field_name=['question', 'text'], max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)

            if not self._buckets:
                pad1 = Pad(max_length=self._max_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)
                pad2 = Pad(max_length=self._max_pair_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {'input_ids': [self._vocab[i] for i in row['question']],
                            "output_ids": [self._vocab[i] for i in row["text"]]}
                    data["input_length"] = len(data["input_ids"])
                    data["output_length"] = len(data["output_ids"])
                    data["input_ids"] = pad1(data["input_ids"])
                    data["output_ids"] = pad2(data["output_ids"])
                    return data
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input_ids": [self._vocab[i] for i in row["question"]],
                            "output_ids": [self._vocab[i] for i in row["text"]]}
                    data["input_length"] = len(data["input_ids"])
                    data["output_length"] = len(data["output_ids"])
                    data["input_ids"] = pad(data["input_ids"])
                    data["output_ids"] = pad(data["output_ids"])
                    if len(data["input_ids"]) > len(data["output_ids"]):
                        data["output_ids"] = Pad.padding(data["output_ids"], len(data["input_ids"]),
                                                         self._vocab.padding_idx)
                    else:
                        data["input_ids"] = Pad.padding(data["input_ids"], len(data["output_ids"]),
                                                        self._vocab.padding_idx)
                    data["padding_length"] = len(data["input_ids"])
                    return data
        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())

            if not self._buckets:
                def token_to_idx(row):
                    model_inputs = self._tokenizer(row["question"], truncation=self._truncation_strategy,
                                                   padding="max_length", max_length=self._max_length)
                    with self._tokenizer.as_target_tokenizer():
                        label = self._tokenizer(row["text"], truncation=self._truncation_strategy,
                                                padding="max_length", max_length=self._max_pair_length)
                    model_inputs["labels"] = label["input_ids"]
                    return model_inputs
            else:
                def token_to_idx(row):
                    document_length = len(self._tokenizer.tokenize(row["question"], add_special_tokens=True))
                    summary_length = len(self._tokenizer.tokenize(row["text"], add_special_tokens=True))
                    d_i = 0
                    for d_i in self._buckets:
                        if d_i >= document_length:
                            break
                    s_i = 0
                    for s_i in self._buckets:
                        if s_i >= summary_length:
                            break
                    i = d_i if d_i > s_i else s_i
                    model_inputs = self._tokenizer(row["question"], truncation=self._truncation_strategy,
                                                   padding="max_length", max_length=i)

                    with self._tokenizer.as_target_tokenizer():
                        label = self._tokenizer(row["text"], truncation=self._truncation_strategy,
                                                padding="max_length", max_length=i)
                    model_inputs["labels"] = label["input_ids"]
                    model_inputs["padding_length"] = len(model_inputs["input_ids"])
                    return model_inputs
        return token_to_idx

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        raise NotImplementedError(f"{self.__class__} should be preprocessed by data stream.")
