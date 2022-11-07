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
    Convai2 dataset
"""
from typing import Union, List, Dict, Optional

import os
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from mindspore.mindrecord import FileWriter

from ..base_dataset import Dataset
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class Convai2Dataset(Dataset):
    """
    Convai2 dataset.

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
        >>> c2 = Convai2Dataset()
        >>> dataset = c2()
    """

    def __init__(self, paths: Optional[str] = None, tokenizer: Union[str] = 'facebook/bart-large', lang: str = None,
                 max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>', **kwargs):
        super(Convai2Dataset, self).__init__(name='Convai2', **kwargs)
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

    def load(self, paths: Optional[str] = None) -> Dict[str, DataFrame]:
        """
        Load Convai2 dataset.

        Args:
            paths (str, Optional): Convai2 dataset directory path, default None.

        Returns:
            Dict[str, DataFrame]: A Convai2 dataset dict.
        """
        if paths is None:
            paths = self.download()
        if not os.path.isdir(paths):
            raise NotADirectoryError(f"{paths} is not a valid directory.")

        files = {"train": "train_self_original_no_cands.txt",
                 "dev": "valid_self_original_no_cands.txt"}

        self._datasets = {}
        for name, filename in files.items():
            filepath = os.path.join(paths, filename)
            if not os.path.isfile(filepath):
                if "test" not in name:
                    raise FileNotFoundError(f"{name} not found in directory {filepath}.")
            self._datasets[name] = self._load(filepath)
        return self._datasets

    def _load(self, path: str) -> DataFrame:
        with open(path, "r", encoding="utf-8") as f:
            length = len(f.readlines())
        with open(path, "r", encoding="utf-8") as f:
            dataset = DataFrame()
            persona = None
            for line in tqdm(f, total=length):
                if "your persona:" in line:
                    epis = False
                    line = line.split(" ")
                    if line[0] == '1':
                        persona = []
                    persona.append(" ".join(line[1:]))
                else:
                    if not epis:
                        epis = True
                        persona = "".join(persona)
                        query, post = line.split("\t")
                        query = " ".join(query.split(" ")[1:])
                        post = post.strip()
                        query = persona + query
                        dataset = dataset.append([[query, post]], ignore_index=True)
                    else:
                        persona = "".join(persona)
                        query_epis, post_epis = line.split("\t")
                        query_epis = " ".join(query_epis.split(" ")[1:])
                        query = query + '\n' + post + '\n' + query_epis
                        post = post_epis.strip()
                        dataset = dataset.append([[query, post]], ignore_index=True)
            dataset.columns = ["query", "post"]
        return dataset

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        if isinstance(self._tokenizer, PreTrainedTokenizerBase):
            self._pretrained_model_inputs_query = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())
            self._pretrained_model_inputs_post = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_pair_length, int)).data.keys())
            query = DataFrame(self.tokenize_progress(dataset, dataset_type, field="query"))
            dataset.drop("query", axis=1, inplace=True)

            temp_pair_length = self._max_pair_length
            temp_length = self._max_length
            self._max_length = self._max_pair_length

            post = DataFrame(self.tokenize_progress(dataset, dataset_type, field="post"))
            dataset.drop("post", axis=1, inplace=True)

            self._max_pair_length = temp_pair_length
            self._max_length = temp_length

            def query_list_split(row):
                data = row["query"]
                return tuple(data)

            query = query.apply(query_list_split, axis=1, result_type="expand")

            def post_list_split(row):
                data = row["post"]
                return tuple(data)

            post = post.apply(post_list_split, axis=1, result_type="expand")
            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                query.columns = self._pretrained_model_inputs_query
                self._max_length = query["length"].max()
                query = DataFrame(
                    self.padding_progress(query, dataset_type, pad_function=self._tokenizer.pad))
                query.columns = self._pretrained_model_inputs_query
                query.drop("length", axis=1, inplace=True)
                self._pretrained_model_inputs_query.remove("length")
            else:
                query.columns = self._pretrained_model_inputs_query

            if not isinstance(self._buckets, List) and not isinstance(self._max_pair_length, int):
                post.columns = self._pretrained_model_inputs_post
                self._max_pair_length = post["length"].max()
                temp_pair_length = self._max_pair_length
                temp_length = self._max_length
                self._max_length = self._max_pair_length
                post = DataFrame(
                    self.padding_progress(post, dataset_type, pad_function=self._tokenizer.pad))
                self._max_pair_length = temp_pair_length
                self._max_length = temp_length
                post.columns = self._pretrained_model_inputs_post
                post.drop("length", axis=1, inplace=True)
                self._pretrained_model_inputs_post.remove("length")
            else:
                post.columns = self._pretrained_model_inputs_post

            dataset[query.columns] = query
            dataset["labels"] = query["input_ids"]
            del query
            del post
            if isinstance(self._buckets, List):
                dataset["input_ids_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
                dataset["labels_length"] = self.get_length_progress(dataset, dataset_type, "labels")
                group = dataset.groupby("input_ids_length")
                for i in group:
                    _, dataset_group = i
                    self._max_length = dataset_group["labels_length"].max()
                    dataset_group = DataFrame(
                        self.padding_progress(DataFrame({"input_ids": dataset_group['labels']}), dataset_type,
                                              pad_function=self._tokenizer.pad))
                    dataset_group.columns = self._pretrained_model_inputs
                    dataset['labels'][dataset_group.index] = dataset_group['input_ids']
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
            self._pretrained_model_inputs = self._pretrained_model_inputs_query
        else:
            raise TypeError("`tokenizer` should be assigned a pretrained tokenizer")
        return dataset

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        if isinstance(self._tokenizer, PreTrainedTokenizerBase):
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())
        else:
            raise TypeError("`tokenizer` should be assigned a pretrained tokenizer")

        if not self._buckets:
            def token_to_idx(row):
                model_inputs = self._tokenizer(row["query"], truncation=self._truncation_strategy,
                                               padding="max_length", max_length=self._max_length)
                with self._tokenizer.as_target_tokenizer():
                    label = self._tokenizer(row["post"], truncation=self._truncation_strategy,
                                            padding="max_length", max_length=self._max_pair_length)
                model_inputs["labels"] = label["input_ids"]
                return model_inputs
        else:
            def token_to_idx(row):
                document_length = len(self._tokenizer.tokenize(row["query"], add_special_tokens=True))
                summary_length = len(self._tokenizer.tokenize(row["post"], add_special_tokens=True))
                d_i = 0
                for d_i in self._buckets:
                    if d_i >= document_length:
                        break
                s_i = 0
                for s_i in self._buckets:
                    if s_i >= summary_length:
                        break
                i = d_i if d_i > s_i else s_i
                model_inputs = self._tokenizer(row["query"], truncation=self._truncation_strategy,
                                               padding="max_length", max_length=i)

                with self._tokenizer.as_target_tokenizer():
                    label = self._tokenizer(row["post"], truncation=self._truncation_strategy,
                                            padding="max_length", max_length=i)
                model_inputs["labels"] = label["input_ids"]
                model_inputs["padding_length"] = len(model_inputs["input_ids"])
                return model_inputs
        return token_to_idx

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]],
                     process_function: callable = None) -> List[str]:
        """
        Write CLSDataset to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (Union[str, Dict[int, str]]): Path of mindrecord file.
            process_function (callable): A function is used to preprocess data.

        Returns:
            List[str]: Dataset field
        """
        if isinstance(file_path, Dict):
            writer = {}
            for k, v in file_path.items():
                writer[k] = FileWriter(file_name=v, shard_num=1)
        else:
            writer = FileWriter(file_name=file_path, shard_num=1)
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            data_schema = {
                'input_ids': {'type': 'int32', 'shape': [-1]},
                'input_length': {'type': 'int32', 'shape': [-1]}}
        else:
            data_schema = {}
            for i in self._pretrained_model_inputs:
                data_schema[i] = {'type': 'int32', 'shape': [-1]}

        if callable(process_function):
            colmun_names = dataset.iterrows()
            i, row = next(colmun_names)
            row = process_function(row)
            if ("labels" in row.keys()) or ("output_ids" in row.keys()):
                if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                    data_schema["output_ids"] = {"type": "int32", "shape": [-1]}
                    data_schema["output_length"] = {"type": "int32", "shape": [-1]}
                else:
                    data_schema["labels"] = {"type": "int32", "shape": [-1]}
        else:
            if ("labels" in dataset.columns.values) or ("output_ids" in dataset.columns.values):
                if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                    data_schema["output_ids"] = {"type": "int32", "shape": [-1]}
                    data_schema["output_length"] = {"type": "int32", "shape": [-1]}
                else:
                    data_schema["labels"] = {"type": "int32", "shape": [-1]}

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
            if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                sample = {'input_ids': np.array(row["input_ids"], dtype=np.int64),
                          'input_length': np.array(row["input_length"], dtype=np.int64)}
            else:
                sample = {}
                for i in self._pretrained_model_inputs:
                    sample[i] = np.array(row[i], dtype=np.int64)

            if callable(process_function):
                if ("labels" in row.keys()) or ("output_ids" in row.keys()):
                    if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                        sample['output_ids'] = np.array(row["output_ids"], dtype=np.int64)
                        sample['output_length'] = np.array(row["output_length"], dtype=np.int64)
                    else:
                        sample['labels'] = np.array(row['labels'], dtype=np.int64)
            else:
                if ("labels" in dataset.columns.values) or ("output_ids" in dataset.columns.values):
                    if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                        sample['output_ids'] = np.array(row["output_ids"], dtype=np.int64)
                        sample['output_length'] = np.array(row["output_length"], dtype=np.int64)
                    else:
                        sample['labels'] = np.array(row['labels'], dtype=np.int64)

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
