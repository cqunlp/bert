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
    conll2003 dataset
"""
import os
from typing import Union, Dict, List, Optional, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import DataFrame

from mindnlp.mindnlp.mindtext.dataset.base_dataset import Dataset
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter

from ..utils import check_loader_paths
from .. import Vocabulary, Pad
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class CONLLDataset(Dataset):
    """
    CONLL Dataset.

    Args:
        paths (Union[str, Dict[str, str]], Optional): Dataset file path or Dataset directory path, default None.
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.
        buckets (List[int], Optional): Padding row to the length of buckets, default None.

    Examples:
       >>> conll = CONLLDataset(paths=Path to the dataset folder)
       >>> dataset = conll()
    """

    def __init__(self, paths: Union[str, Dict[str, str]] = None, max_size: int = None,
                 min_freq: int = None, padding: str = '<pad>', unknown: str = '<unk>',
                 buckets: List[int] = None, **kwargs):
        super(CONLLDataset, self).__init__(sep='\t', name='conll', **kwargs)
        self._paths = paths
        self._tokenize = None
        self._lang = None
        self._pos_tags = None
        self._chunk_tags = None
        self._ner_tags = None
        self._vocab_max_size = max_size
        self._vocab_min_freq = min_freq
        self._padding = padding
        self._unknown = unknown
        self._buckets = buckets

    def __call__(self) -> Dict[str, ds.MindDataset]:
        self.load(self._paths)
        self.process(tokenizer=self._tokenize, lang=self._lang, max_size=self._vocab_max_size,
                     min_freq=self._vocab_min_freq, padding=self._padding,
                     unknown=self._unknown, buckets=self._buckets)
        return self._mind_datasets

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> Dict[str, DataFrame]:
        files = {'chunk_tags': 'chunk_tags.csv',
                 'ner_tags': 'ner_tags.csv',
                 'pos_tags': 'pos_tags.csv'}

        self._pos_tags = [j for i in pd.read_csv(os.path.join(paths, files['pos_tags'])).values.tolist() for j in i]
        self._ner_tags = [j for i in pd.read_csv(os.path.join(paths, files['ner_tags'])).values.tolist() for j in i]
        self._chunk_tags = [j for i in pd.read_csv(os.path.join(paths, files['chunk_tags'])).values.tolist() for j in i]
        if not paths:
            paths = self.download()
        paths = check_loader_paths(paths)
        self._datasets = {name: self._load(path) for name, path in paths.items()}

        vocab = Vocabulary(max_size=self._vocab_max_size, min_freq=self._vocab_min_freq, padding=self._padding,
                           unknown=self._unknown)
        for i in self._datasets.keys():
            vocab_bar = tqdm(self._datasets[i][['tokens']].iterrows(), total=len(self._datasets[i]))
            for _, row in vocab_bar:
                vocab.update([i.lower() for i in row['tokens']])
                vocab_bar.set_description("Build Vocabulary")
        vocab.build_vocab()
        self._vocab = vocab
        return self._datasets

    def _load(self, path: str) -> DataFrame:
        """
        Load dataset from Conll file.
        """
        dataset = pd.read_csv(path, keep_default_na=False)

        def split_str2liststr(x):
            # $%$ is separator
            return [i for i in x.split("$%$")]

        def split_str2listint(x):
            # $%$ is separator
            return [int(i) for i in x.split("$%$")]

        def convert_to_lower(x):
            return [i.lower() for i in x]

        dataset['tokens'] = dataset['tokens'].map(split_str2liststr)
        dataset['tokens'] = dataset['tokens'].map(convert_to_lower)
        dataset['pos_tags'] = dataset['pos_tags'].map(split_str2listint)
        dataset['chunk_tags'] = dataset['chunk_tags'].map(split_str2listint)
        dataset['ner_tags'] = dataset['ner_tags'].map(split_str2listint)

        return dataset

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        """
        Preprocess dataset.
        """
        dataset["input_ids"] = self._vocab.word_to_idx(dataset["tokens"])
        dataset.drop("tokens", axis=1, inplace=True)
        dataset["input_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
        if not self._buckets:
            if isinstance(self._max_length, int):
                max_length = self._max_length
            else:
                max_length = dataset["input_length"].max()
            pad = Pad(max_length=max_length, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
        else:
            pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)
        dataset["input_ids"] = self.padding_progress(dataset, dataset_type, field="input_ids", pad_function=pad)
        dataset["pos_tags"] = self.padding_progress(dataset, dataset_type, field="pos_tags", pad_function=pad)
        dataset["chunk_tags"] = self.padding_progress(dataset, dataset_type, field="chunk_tags", pad_function=pad)
        dataset["ner_tags"] = self.padding_progress(dataset, dataset_type, field="ner_tags", pad_function=pad)
        if isinstance(self._buckets, List):
            dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
        return dataset

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """Preprocess dataset by data stream."""
        if not self._buckets:
            pad = Pad(max_length=self._max_length, pad_val=self._vocab.padding_idx,
                      truncate=self._truncation_strategy)
        else:
            pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)

        def token_to_idx(row):
            data = {"input_ids": [self._vocab[i] for i in row["tokens"]]}
            data["input_length"] = len(data["input_ids"])
            data["input_ids"] = pad(data["input_ids"])
            if isinstance(self._buckets, List):
                data["padding_length"] = len(data["input_ids"])
            data["pos_tags"] = pad(data["input_ids"])
            data["chunk_tags"] = pad(data["input_ids"])
            data["ner_tags"] = pad(data["input_ids"])
            return data

        return token_to_idx

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]],
                     process_function: callable = None) -> List[str]:
        """
        Write to .mindrecord file.
        """
        if isinstance(file_path, Dict):
            writer = {}
            for k, v in file_path.items():
                writer[k] = FileWriter(file_name=v, shard_num=1)
        else:
            writer = FileWriter(file_name=file_path, shard_num=1)

        data_schema = {
            "input_ids": {"type": "int32", "shape": [-1]},
            "input_length": {"type": "int32", "shape": [-1]},
            "input_mask": {"type": "int32", "shape": [-1]},
            "ner_tags": {"type": "int32", "shape": [-1]},
            "chunk_tags": {"type": "int32", "shape": [-1]},
            "pos_tags": {"type": "int32", "shape": [-1]},
        }

        if isinstance(writer, Dict):
            for k in file_path.keys():
                writer[k].add_schema(data_schema, self._name)
        else:
            writer.add_schema(data_schema, self._name)

        if not isinstance(writer, Dict):
            data = []
        vocab_bar = tqdm(dataset.iterrows(), total=len(dataset))

        for index, row in vocab_bar:
            # Whether using a pretrained model tokenizer.
            if callable(process_function):
                row = process_function(row)
            input_ids = np.array(row["input_ids"], dtype=np.int32)
            sample = {"input_ids": input_ids,
                      "input_length": np.array(row["input_length"], dtype=np.int32),
                      "input_mask": to_mask(np.array(row["input_length"], dtype=np.int32), input_ids.shape),
                      "ner_tags": np.array(row["ner_tags"], dtype=np.int32),
                      "chunk_tags": np.array(row["chunk_tags"], dtype=np.int32),
                      "pos_tags": np.array(row["pos_tags"], dtype=np.int32)}

            if not isinstance(writer, Dict):
                data.append(sample)
                if index % 10 == 0:
                    writer.write_raw_data(data)
                    data = []
            else:
                if row["padding_length"] > list(writer.keys())[-1]:
                    writer[list(writer.keys())[-1]].write_raw_data([sample])
                else:
                    writer[row["padding_length"]].write_raw_data([sample])
            vocab_bar.set_description("Writing data to .mindrecord file")

        if not isinstance(writer, Dict):
            if data:
                writer.write_raw_data(data)
        if not isinstance(writer, Dict):
            writer.commit()
        else:
            for v in writer.values():
                v.commit()
        return list(data_schema.keys())


def to_mask(input_length: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    input_mask = np.zeros(shape)
    input_mask[:input_length] = 1
    return input_mask
