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
    Base Dataset
"""
import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, BatchEncoding
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as deC
from mindspore.mindrecord import FileWriter

from . import Vocabulary, Pad
from .utils import get_cache_path, _get_dataset_url, cached_path, check_loader_paths, get_tokenizer, \
    _preprocess_sequentially, _get_dataset_type, get_split_func

logging.basicConfig(level=logging.NOTSET)


class Dataset:
    """
    Base class of Dataset.

    Dataset supports the following five functions.
        - download(): Download dataset to default path:default path: ~/.mindtext/datasets/.
                the function will return the cache path of the downloaded file.
        - _load(): Read data from a data file, return :class:`panadas.DataFrame`.
        - load(): The files are read separately as DataFrame and are put into a dict.
        - _process(): Preprocess dataset.
        - process(): Preprocess dataset by a tokenizer.
        - _write_to_mr(): Convert dataset to mindreord format.

    Args:
        vocab (Vocabulary, Optional): Convert tokens to index,default None.
        name (str, Optional): Dataset name,default None.
        label_map (Dict[str, int], Optional): Dataset label map,default None.
        batch_size (int): Dataset mini batch size. Refers to `mindspore.dataset.MindDataset`.
        repeat_dataset (int): Dataset repeat numbers. Refers to `mindspore.dataset.MindDataset`.
        num_parallel_workers (int): The number of readers. Refers to `mindspore.dataset.MindDataset`.
        columns_list (List[str]): The columns name of `mindspore.dataset.MindDataset` (train set).
        test_columns_list (List[str]): The columns name of `mindspore.dataset.MindDataset` (test set).
        truncation_strategy (Union[bool, str]): Truncation strategy used to index sequence.When not using a pretrained
            tokenizer, it can be assigned to `True`.When using a pretrained tokenizer, it can be assigned to `True` or
            `longest_first`, `only_first`, `only_second`, `False` or `do_not_truncate`
            (Refers to transformers.PreTrainedTokenizerBase).
        max_length (int): The max length of padding or truncate. Refers to transformers.PreTrainedTokenizerBase
        max_pair_length (int): The max pair length can be assigned when second sequence need to be padded or truncated.
            Refers to transformers.PreTrainedTokenizerBase
        stream (callable): Whether to convert dataset to MindRecord file by data stream.
        process4transformer (bool): Preprocess data for transformer model.
        vocab_file (str): Vocabulary file path. It's used to initialize a `Vocabulary` class when `process4transformer`
            is `True`.
    """

    def __init__(self, vocab: Optional[Vocabulary] = None, name: Optional[str] = None,
                 label_map: Optional[Dict[str, int]] = None, **kwargs):
        self._vocab = vocab
        self._name = name
        self._label_map = label_map
        self._datasets = None
        self._tokenizer = None
        self._mind_datasets = {}
        self._buckets = kwargs.pop("buckets", None)
        self._batch_size = kwargs.pop("batch_size", 8)
        self._repeat_dataset = kwargs.pop("repeat_dataset", 1)
        self._num_parallel_workers = kwargs.pop("num_parallel_workers", None)
        self._columns_list = kwargs.pop("columns_list", None)
        self._test_columns_list = kwargs.pop("test_columns_list", None)
        self._truncation_strategy = kwargs.pop("truncation_strategy", False)
        self._max_length = kwargs.pop("max_length", None)
        self._max_pair_length = kwargs.pop("max_pair_length", None)
        self._stream = kwargs.pop("stream", False)
        self._process4transformer = kwargs.pop("process4transformer", False)
        self._vocab_file = kwargs.pop("vocab_file", None)
        if isinstance(self._max_pair_length, int) and isinstance(self, CLSBaseDataset):
            raise TypeError("`CLSBaseDataset` do not need `max_pair_length`.")
        if (isinstance(self._max_length, int) or isinstance(self._max_pair_length, int)) and isinstance(self._buckets,
                                                                                                        List):
            raise TypeError("`max_length`(or `max_pair_length`) and `buckets` cannot be assigned at the same time.")
        if self._stream:
            if isinstance(self, CLSBaseDataset):
                if not isinstance(self._max_length, int) and not isinstance(self._buckets, List):
                    raise TypeError("`max_length` or `buckets` should be assigned when `stream` is `True`.")
            if isinstance(self, PairCLSBaseDataset):
                if not (isinstance(self._max_length, int) and isinstance(self._max_length, int)) and not isinstance(
                        self._buckets, List):
                    raise TypeError(
                        "`max_length`, `max_pair_length` or `buckets` should be assigned when `stream` is `True`.")
        if isinstance(self, CLSBaseDataset):
            if bool(self._truncation_strategy) and not isinstance(self._max_length, int) and not isinstance(
                    self._buckets, List):
                raise TypeError("`truncation_strategy` need be `False` when `max_length` is not assigned.")
        if isinstance(self, PairCLSBaseDataset):
            if bool(self._truncation_strategy) and not (
                    isinstance(self._max_length, int) or isinstance(self._max_pair_length, int)) and not isinstance(
                        self._buckets, List):
                raise TypeError(
                    "`truncation_strategy` need be `False` when `max_length` or `max_pair_length` is not assigned.")

    def from_cache(self, columns_list: List[str], test_columns_list: List[str], repeat_dataset: int = 1,
                   batch_size: int = 8, num_parallel_workers: Optional[int] = None) -> Dict[str, ds.MindDataset]:
        """
        Read dataset from cache.

        Args:
            columns_list (List[str]): Train or dev dataset columns list.
            test_columns_list (List[str]): Test dataset columns list.
            repeat_dataset (int): Repeat dataset.
            batch_size (int): Batch size.
            num_parallel_workers (int, Optional): The number of readers.

        Returns:
            Dict[str, ds.MindDataset]: A MindDataset dictionary.
        """
        mr_dir_path = Path(get_cache_path()) / Path("dataset") / Path(self._name).joinpath("mindrecord")
        data_path_list = os.listdir(mr_dir_path)
        if not mr_dir_path.exists() or os.listdir(mr_dir_path) == 0:
            raise FileNotFoundError(f"{self._name} dataset not founded in cache.")
        for i in data_path_list:
            if "test" in i:
                columns = test_columns_list
            else:
                columns = columns_list
            data_path = mr_dir_path.joinpath(i)
            index = 0
            for file in glob.glob(str(data_path.joinpath("*.mindrecord"))):
                per_bucket_dataset = ds.MindDataset(dataset_files=str(data_path.joinpath(file)),
                                                    columns_list=columns,
                                                    num_parallel_workers=num_parallel_workers)
                type_cast_op = deC.TypeCast(mstype.int32)
                for name in columns:
                    per_bucket_dataset = per_bucket_dataset.map(operations=type_cast_op, input_columns=name)
                per_bucket_dataset = per_bucket_dataset.batch(batch_size, drop_remainder=False)
                per_bucket_dataset = per_bucket_dataset.repeat(repeat_dataset)
                if index == 0:
                    if per_bucket_dataset.get_dataset_size() != 0:
                        self._mind_datasets[i] = per_bucket_dataset
                else:
                    if per_bucket_dataset.get_dataset_size() != 0:
                        self._mind_datasets[i] += per_bucket_dataset
                index += 1
            self._mind_datasets[i] = self._mind_datasets[i].shuffle(self._mind_datasets[i].get_dataset_size())
        return self._mind_datasets

    def _load(self, path: str) -> DataFrame:
        """
        Given a path, return the DataFrame.

        Args:
            path (str): Dataset file path.

        Returns:
            DataFrame: Dataset file will be read as a DataFrame.
        """
        with open(path, "r", encoding="utf-8") as f:
            columns = f.readline().strip().split(self.sep)
            dataset = pd.read_csv(f, sep="\n", names=columns)
            tqdm.pandas(desc=f"{self._name} dataset loadding")
            split_row = get_split_func(dataset, self.sep)
            dataset = dataset.progress_apply(split_row, axis=1, result_type="expand")
            dataset.columns = columns
        dataset.fillna("")
        dataset.dropna(inplace=True)
        return dataset

    def load(self, paths: Optional[Union[str, Dict[str, str]]] = None) -> Dict[str, DataFrame]:
        """
        Read data from a file in one or more specified paths.

        Args:
            paths (Union[str, Dict[str, str]], Optional): Dataset path, default None.

        Returns:
            Dict[str, DataFrame]: A Dataset dictionary.

        Examples::
            There are several inputs mode:
            0.If None, it checks to see if there is a local cache. If not, it is automatically downloaded and cached.
            1.given a directory, the "train" in directory will be considered to be train dataset::
                ds = xxxDataset()
                dataset = ds.load("/path/dir")
                #  dataset = {"train":..., "dev":..., "test":...} if the directory contains "train", "dev", "test".
            2.given a dict,such as train,dev,test not in the same directory,or the train, dev, test are not contained
            in directory::
                paths = {"train":"/path/to/train.tsv", "dev":"/to/validate.tsv", "test":"/to/test.tsv"}
                ds = xxxDataset()
                dataset = ds.load(paths)
                #  dataset = {"train":..., "dev":..., "test":...}
            3.give a file name::
                ds = xxxDataset()
                dataset = ds.load("/path/to/a/train.conll")
        """
        if not paths:
            paths = self.download()
        paths = check_loader_paths(paths)
        self._datasets = {name: self._load(path) for name, path in paths.items()}
        return self._datasets

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """
        Preprocess dataset by data stream.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str):  Dataset type(train, dev, test).
                Different types of datasets may be processed differently.

        Returns:
            callable: A preprocess function.
        """
        raise NotImplementedError(f"{self.__class__} cannot be preprocessed by data stream.")

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        """
        Preprocess dataset.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str): Dataset type(train, dev, test).

        Returns:
            DataFrame: Preprocessed dataset.
        """
        raise NotImplementedError(f"{self.__class__} cannot be preprocessed.")

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
        if isinstance(self, (CLSBaseDataset, PairCLSBaseDataset)):
            if isinstance(self._tokenizer, PreTrainedTokenizerBase) and isinstance(self._max_pair_length, int):
                raise TypeError("`max_pair_length` cannot be assigned when use a pretrained tokenizer.")
        dataset_file_name = _preprocess_sequentially(list(self._datasets.keys()))

        for dataset_name in dataset_file_name:
            dataset = self._datasets.get(dataset_name)
            d_t = _get_dataset_type(dataset_name)
            if isinstance(dataset, DataFrame):
                if self._stream:
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

    def convert_to_mr(self, dataset: DataFrame, file_name: str, is_test: bool,
                      process_function: callable = None) -> ds.MindDataset:
        """
        Convert dataset to .mindrecord format file,and read as MindDataset.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_name (str): Name of .mindrecord file.
            is_test (bool): Whether the data set is a test set.
            process_function (callable): A function is used to preprocess data.

        Returns:
            MindDataset: A MindDataset.
        """
        mr_dir_path = Path(get_cache_path()) / Path("dataset") / Path(self._name).joinpath("mindrecord", file_name)
        if not mr_dir_path.exists():
            mr_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(mr_dir_path)
            mr_dir_path.mkdir(parents=True, exist_ok=True)
        md_dataset = None
        type_cast_op = deC.TypeCast(mstype.int32)
        if not self._buckets:
            file_path = mr_dir_path.joinpath(file_name + ".mindrecord")
            field_name = self._write_to_mr(dataset, str(file_path), process_function=process_function)
            if is_test:
                if isinstance(self._test_columns_list, List):
                    field_name = self._test_columns_list
            else:
                if isinstance(self._columns_list, List):
                    field_name = self._columns_list
            md_dataset = ds.MindDataset(dataset_files=str(file_path), columns_list=field_name)
            for name in field_name:
                md_dataset = md_dataset.map(operations=type_cast_op, input_columns=name)
            md_dataset = md_dataset.batch(self._batch_size, drop_remainder=False)
            md_dataset = md_dataset.repeat(self._repeat_dataset)
        else:
            if self._stream:
                file_paths = {}
                for i in range(len(self._buckets)):
                    file_paths[self._buckets[i]] = str(mr_dir_path.joinpath(
                        file_name + "_" + str(self._buckets[i]) + ".mindrecord"))
                field_name = self._write_to_mr(dataset, file_paths, process_function=process_function)
                if is_test:
                    if isinstance(self._test_columns_list, List):
                        field_name = self._test_columns_list
                else:
                    if isinstance(self._columns_list, List):
                        field_name = self._columns_list
                for path in file_paths.values():
                    per_bucket_dataset = ds.MindDataset(dataset_files=str(path), columns_list=field_name)
                    for name in field_name:
                        per_bucket_dataset = per_bucket_dataset.map(operations=type_cast_op, input_columns=name)
                    per_bucket_dataset = per_bucket_dataset.batch(self._batch_size, drop_remainder=False)
                    per_bucket_dataset = per_bucket_dataset.repeat(self._repeat_dataset)
                    if not md_dataset:
                        if per_bucket_dataset.get_dataset_size() != 0:
                            md_dataset = per_bucket_dataset
                    else:
                        if per_bucket_dataset.get_dataset_size() != 0:
                            md_dataset += per_bucket_dataset
            else:
                for i in range(len(self._buckets)):
                    file_path = mr_dir_path.joinpath(file_name + "_" + str(self._buckets[i]) + ".mindrecord")
                    if i == len(self._buckets) - 1:
                        dataset_bucket = dataset[dataset["padding_length"] >= self._buckets[i]]
                    else:
                        dataset_bucket = dataset[dataset["padding_length"] == self._buckets[i]]
                    if not dataset_bucket.index.empty:
                        field_name = self._write_to_mr(dataset_bucket, str(file_path), is_test)
                        if is_test and isinstance(self._test_columns_list, List):
                            field_name = self._test_columns_list
                        elif not is_test and isinstance(self._columns_list, List):
                            field_name = self._columns_list
                        per_bucket_dataset = ds.MindDataset(dataset_files=str(file_path), columns_list=field_name)
                        for name in field_name:
                            per_bucket_dataset = per_bucket_dataset.map(operations=type_cast_op, input_columns=name)
                        per_bucket_dataset = per_bucket_dataset.batch(self._batch_size, drop_remainder=False)
                        per_bucket_dataset = per_bucket_dataset.repeat(self._repeat_dataset)
                        if not md_dataset:
                            md_dataset = per_bucket_dataset
                        else:
                            md_dataset += per_bucket_dataset
        md_dataset = md_dataset.shuffle(md_dataset.get_dataset_size())
        return md_dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]],
                     process_function: callable = None) -> List[str]:
        """
        Write to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (Union[str, Dict[int, str]]): Path of mindrecord file.
            process_function (callable): A function is used to preprocess data.

        Returns:
            List[str]: Dataset field.
        """
        raise NotImplementedError

    @staticmethod
    def _get_dataset_path(dataset_name: str) -> Union[str, Path]:
        """
        Given a dataset name, try to read the dataset directory, if not exits,
        the function will try to download the corresponding dataset.

        Args:
            dataset_name (str): Dataset name.

        Returns:
             Union[str, Path]: Dataset directory path.
        """
        default_cache_path = get_cache_path()
        url = _get_dataset_url(dataset_name)
        output_dir = cached_path(url_or_filename=[dataset_name, url], cache_dir=default_cache_path, name="dataset")

        return output_dir

    @property
    def vocab(self) -> Vocabulary:
        """
        Return vocabulary.

        Returns:
            Vocabulary: Dataset Vocabulary.
        """
        return self._vocab

    @property
    def datasets(self) -> Dict[str, ds.MindDataset]:
        """
        Return mindDataset.

        Returns:
            Dict[str, MindDataset]: A dict of mindDataset.
        """
        return self._mind_datasets

    def label_to_idx(self, row: str) -> int:
        """
        Convert label from a token to index.

        Args:
            row (str): Label tokens.

        Returns:
            str: Label index.
        """
        return self._label_map[row]

    def _pretrained_tokenize(self, row: Union[str, pd.Series]) -> List:
        """
        Tokenize data by pretrained tokenizer.

        Args:
            row (Union[str, pd.Series]): Dataset row.

        Returns:
            List: A tokenized data.
        """
        if isinstance(self._buckets, List):
            if isinstance(self, PairCLSBaseDataset):
                length = len(
                    self._tokenizer.tokenize(row["sentence1"], row["sentence2"], add_special_tokens=True))
            else:
                length = len(self._tokenizer.tokenize(row, add_special_tokens=True))
            i = 0
            for i in self._buckets:
                if i >= length:
                    break
            if isinstance(self, PairCLSBaseDataset):
                data = self._tokenizer(row["sentence1"], row["sentence2"],
                                       truncation=self._truncation_strategy, padding="max_length",
                                       max_length=i)
            else:
                data = self._tokenizer(row, truncation=self._truncation_strategy, padding="max_length",
                                       max_length=i)
        else:
            if isinstance(self, PairCLSBaseDataset):
                if isinstance(self._max_length, int):
                    data = self._tokenizer(row["sentence1"], row["sentence2"], padding="max_length",
                                           truncation=self._truncation_strategy, max_length=self._max_length)
                else:
                    data = self._tokenizer(row["sentence1"], row["sentence2"], return_length=True)
                    data["length"] = data["length"][0]
            else:
                if isinstance(self._max_length, int):
                    data = self._tokenizer(row, padding="max_length", truncation=self._truncation_strategy,
                                           max_length=self._max_length)
                else:
                    data = self._tokenizer(row, return_length=True)
                    data["length"] = data["length"][0]
        data = [v for k, v in data.items()]
        return data

    def tokenize_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str, field: Union[str, List[str]]) \
            -> Union[DataFrame, pd.Series]:
        """
        Tokenizer with progress bar.

        Args:
            dataset (Union[DataFrame, Series]): Data need to be tokenized.
            dataset_type (str): Dataset type(train, dev, test).
            field (str): Field name.

        Returns:
            Union[DataFrame, Series]: Tokenized data.
        """
        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(tokenize).")
        if isinstance(self._tokenizer, PreTrainedTokenizerBase):
            if isinstance(self, PairCLSBaseDataset):
                dataset = dataset[field].progress_apply(self._pretrained_tokenize, axis=1, result_type="expand")
            else:
                dataset = dataset[field].progress_apply(self._pretrained_tokenize)
        else:
            tokenizer = self._tokenizer
            dataset = dataset[field].progress_apply(tokenizer)
        return dataset

    def get_length_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str, field: str) -> Union[
            DataFrame, pd.Series]:
        """
        Get sentence length.

        Args:
            dataset (Union[DataFrame, Series]): Data need to be processed.
            dataset_type (str): Dataset type(train, dev, test).
            field (str): Field name.

        Returns:
            Union[DataFrame, Series]: Processed data.
        """
        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(length).")
        return dataset[field].progress_apply(len)

    def _create_padding_function(self, pad_function: PreTrainedTokenizerBase.pad):
        """
        Create a padding function by pretrained tokenizer.

        Args:
            pad_function (PreTrainedTokenizerBase.pad):

        Returns:
            callable: Padding function.
        """

        def pad_func(row: pd.Series):
            inputs = {}
            for i in row.keys():
                inputs[i] = row[i]
            data = pad_function(BatchEncoding(inputs), padding="max_length", max_length=self._max_length)
            data = [v for k, v in data.items()]
            return data

        return pad_func

    def padding_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str,
                         pad_function: Union[Pad, PreTrainedTokenizerBase.pad], field: Optional[str] = None) -> Union[
                             DataFrame, pd.Series]:
        """
        Padding index sequence.

        Args:
            dataset (Union[DataFrame, Series]): Data need to padding.
            dataset_type (str): Dataset type(train, dev, test).
            field (str, Optional): Field name.
            pad_function (Union[Pad, PreTrainedTokenizerBase.pad]): Pad class or a pretrained tokenizer pad function.

        Returns:
            Union[DataFrame, Series]: Processed data.
        """
        if isinstance(self._tokenizer, PreTrainedTokenizerBase):
            tqdm.pandas(desc=f"{self._name} {dataset_type} dataset preprocess bar(padding).")
            dataset = dataset.progress_apply(self._create_padding_function(pad_function), axis=1, result_type="expand")
        else:
            tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(padding).")
            dataset = dataset[field].progress_apply(pad_function)
        return dataset

    def padding_same_progress(self, dataset: Union[DataFrame, pd.Series], dataset_type: str,
                              field: Union[str, List[str]], pad_val: Optional[int] = None) -> Union[
                                  DataFrame, pd.Series]:
        """
        Pad both sentences to the same length.

        Args:
            dataset (Union[DataFrame, Series]): Data need to padding.
            dataset_type (str): Dataset type(train, dev, test).
            field (Union[str, List[str]]): Field name.
            pad_val (Optional[int]): Padding value.

        Returns:
            Union[DataFrame, Series]: Processed data.
        """
        if isinstance(field, List):
            input1_name = field[0]
            input2_name = field[1]

        pad_val = pad_val if isinstance(pad_val, int) else self._vocab.padding_idx

        def padding_same(row):
            if len(row[input1_name]) > len(row[input2_name]):
                input2 = Pad.padding(row[input1_name], len(row[input2_name]), pad_val)
                re = row[input1_name], input2
            else:
                input1 = Pad.padding(row[input1_name], len(row[input2_name]), pad_val)
                re = input1, row[input2_name]
            return re

        tqdm.pandas(desc=f"{self._name} {dataset_type} dataset {field} preprocess bar(same padding).")
        dataset = dataset[field].progress_apply(padding_same, axis=1,
                                                result_type="expand")
        dataset.columns = field
        return dataset

    def download(self) -> str:
        """
        Dataset download.

        Returns:
            str: The downloaded dataset directory.
        """
        output_dir = self._get_dataset_path(dataset_name=self._name)
        return output_dir

    def __getitem__(self, dataset_type: str) -> DataFrame:
        """
        Return dataset by dataset_type.

        Args:
            dataset_type (str): Dataset type.

        Returns:
            DataFrame: Dataset(train, dev, test).
        """
        return self._mind_datasets[dataset_type]

    def __str__(self) -> str:
        return str(
            dict(zip(self._mind_datasets.keys(), [value.get_dataset_size() for value in self._mind_datasets.values()])))


class CLSBaseDataset(Dataset):
    """
    A base class of text classification.

    Args:
        sep (str): The separator for pandas reading file, default `,`.
    """

    def __init__(self, sep: str = ",", **kwargs):
        super(CLSBaseDataset, self).__init__(**kwargs)
        self.sep = sep
        self._label_nums = None

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """
        Preprocess dataset by data stream.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str):  Dataset type(train, dev, test).
                Different types of datasets may be processed differently.

        Returns:
            callable: A preprocess function.
        """
        if "label" in dataset.columns.values:
            if isinstance(self._label_map, Dict):
                dataset["label"] = dataset["label"].map(self.label_to_idx)
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["sentence"] = self.tokenize_progress(dataset, dataset_type, "sentence")

            if dataset_type == "train":
                self._label_nums = dataset["label"].value_counts().shape[0]
                self._vocab = Vocabulary.from_dataset(dataset, field_name="sentence", max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)

            if not self._buckets:
                pad = Pad(max_length=self._max_length, pad_val=self._vocab.padding_idx,
                          truncate=self._truncation_strategy)
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)

            def token_to_idx(row):
                data = {"input_ids": [self._vocab[i] for i in row["sentence"]]}
                data["input_length"] = len(data["input_ids"])
                data["input_ids"] = pad(data["input_ids"])
                if isinstance(self._buckets, List):
                    data["padding_length"] = len(data["input_ids"])
                if "label" in row.keys():
                    data["label"] = row["label"]
                return data
        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())

            def token_to_idx(row):
                data = {}
                tokenized_data = self._pretrained_tokenize(row["sentence"])
                for k, v in zip(self._pretrained_model_inputs, tokenized_data):
                    data[k] = v
                if isinstance(self._buckets, List):
                    data["padding_length"] = len(data["input_ids"])
                if "label" in row.keys():
                    data["label"] = row["label"]
                return data
        return token_to_idx

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        """
        Classification dataset preprocess function.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str): Dataset type(train, dev, test).
                Different types of datasets may be processed differently.

        Returns:
            DataFrame: Preprocessed dataset.
        """
        if "label" in dataset.columns.values:
            if isinstance(self._label_map, Dict):
                dataset["label"] = dataset["label"].map(self.label_to_idx)
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["sentence"] = self.tokenize_progress(dataset, dataset_type, "sentence")

            if dataset_type == "train":
                self._label_nums = dataset["label"].value_counts().shape[0]
                self._vocab = Vocabulary.from_dataset(dataset, field_name="sentence", max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)
            dataset["input_ids"] = self._vocab.word_to_idx(dataset["sentence"])
            dataset.drop("sentence", axis=1, inplace=True)
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
            if isinstance(self._buckets, List):
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())
            dataset_tokenized = DataFrame(self.tokenize_progress(dataset, dataset_type, field="sentence"))
            dataset.drop("sentence", axis=1, inplace=True)

            def _list_split(row):
                data = row["sentence"]
                return tuple(data)

            tqdm.pandas(desc=f"{self._name} {dataset_type} dataset processing.")
            dataset_tokenized = dataset_tokenized.progress_apply(_list_split, axis=1, result_type="expand")
            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                dataset_tokenized.columns = self._pretrained_model_inputs
                self._max_length = dataset_tokenized["length"].max()
                dataset_tokenized = DataFrame(
                    self.padding_progress(dataset_tokenized, dataset_type, pad_function=self._tokenizer.pad))
            dataset_tokenized.columns = self._pretrained_model_inputs
            if isinstance(self._buckets, List):
                dataset_tokenized["padding_length"] = self.get_length_progress(dataset_tokenized, dataset_type,
                                                                               "input_ids")
            if "label" in dataset.columns.values:
                dataset_tokenized["label"] = dataset["label"]
            dataset = dataset_tokenized
            del dataset_tokenized
            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                self._pretrained_model_inputs.remove("length")
        return dataset

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
                "input_ids": {"type": "int32", "shape": [-1]},
                "input_length": {"type": "int32", "shape": [-1]}}
        else:
            data_schema = {}
            for i in self._pretrained_model_inputs:
                data_schema[i] = {"type": "int32", "shape": [-1]}

        if callable(process_function):
            colmun_names = dataset.iterrows()
            i, row = next(colmun_names)
            row = process_function(row)
            if "label" in row.keys():
                data_schema["label"] = {"type": "int32", "shape": [-1]}
        else:
            if "label" in dataset.columns.values:
                data_schema["label"] = {"type": "int32", "shape": [-1]}

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
                sample = {"input_ids": np.array(row["input_ids"], dtype=np.int32),
                          "input_length": np.array(row["input_length"], dtype=np.int32)}
            else:
                sample = {}
                for i in self._pretrained_model_inputs:
                    sample[i] = np.array(row[i], dtype=np.int32)

            if callable(process_function):
                if "label" in row.keys():
                    sample["label"] = np.array(row["label"], dtype=np.int32)
            else:
                if "label" in dataset.columns.values:
                    sample["label"] = np.array(row["label"], dtype=np.int32)

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

    @property
    def label_nums(self) -> int:
        """
        Return label_nums.
        """
        return self._label_nums

    @label_nums.setter
    def label_nums(self, nums: int):
        """
        Need to be assigned.

        Args:
            nums (str): The number of label.
        """
        self._label_nums = nums


class PairCLSBaseDataset(Dataset):
    """
    A base class of  pair text classification.

    Args:
        sep (str): The separator for pandas reading file, default `,`.
        label_is_float (bool): Whether the label of the dataset is float, default False.
    """

    def __init__(self, sep: str = ",", label_is_float: bool = False, **kwargs):
        super(PairCLSBaseDataset, self).__init__(**kwargs)
        self.sep = sep
        self._label_is_float = label_is_float
        self._label_nums = None

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """
        Preprocess dataset by data stream.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str):  Dataset type(train, dev, test).
                Different types of datasets may be processed differently.

        Returns:
            callable: A preprocess function.
        """
        if "label" in dataset.columns.values:
            if not self._label_is_float and isinstance(self._label_map, Dict):
                dataset["label"] = dataset["label"].map(self.label_to_idx)
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["sentence1"] = self.tokenize_progress(dataset, dataset_type, "sentence1")
            dataset["sentence2"] = self.tokenize_progress(dataset, dataset_type, "sentence2")

            if dataset_type == "train":
                self._label_nums = dataset["label"].value_counts().shape[0]
                self._vocab = Vocabulary.from_dataset(dataset, field_name=["sentence1", "sentence2"], max_size=max_size,
                                                      min_freq=min_freq, padding=padding, unknown=unknown)

            if not self._buckets:
                pad1 = Pad(max_length=self._max_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)
                pad2 = Pad(max_length=self._max_pair_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input1_ids": [self._vocab[i] for i in row["sentence1"]],
                            "input2_ids": [self._vocab[i] for i in row["sentence2"]]}
                    data["input1_length"] = len(data['input1_ids'])
                    data["input2_length"] = len(data['input2_ids'])
                    data["input1_ids"] = pad1(data["input1_ids"])
                    data["input2_ids"] = pad2(data["input2_ids"])
                    if "label" in row.keys():
                        data["label"] = row["label"]
                    return data
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input1_ids": [self._vocab[i] for i in row["sentence1"]],
                            "input2_ids": [self._vocab[i] for i in row["sentence2"]]}
                    data["input1_length"] = len(data['input1_ids'])
                    data["input2_length"] = len(data['input2_ids'])
                    data["input1_ids"] = pad(data["input1_ids"])
                    data["input2_ids"] = pad(data["input2_ids"])
                    if len(data["input1_ids"]) > len(data["input2_ids"]):
                        data["input2_ids"] = Pad.padding(data["input2_ids"], len(data["input1_ids"]),
                                                         self._vocab.padding_idx)
                    else:
                        data["input1_ids"] = Pad.padding(data["input1_ids"], len(data["input2_ids"]),
                                                         self._vocab.padding_idx)
                    data["padding_length"] = len(data["input1_ids"])
                    if "label" in row.keys():
                        data["label"] = row["label"]
                    return data
        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())

            def token_to_idx(row):
                data = {}
                tokenized_data = self._pretrained_tokenize(row)
                for k, v in zip(self._pretrained_model_inputs, tokenized_data):
                    data[k] = v
                if isinstance(self._buckets, List):
                    data["padding_length"] = len(data["input_ids"])
                if "label" in row.keys():
                    data["label"] = row["label"]
                return data
        return token_to_idx

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                 dataset_type: str) -> DataFrame:
        """
        Pair text classification dataset preprocess function.

        Args:
           dataset (DataFrame): DataFrame need to preprocess.
           max_size (int): Vocab max size.
           min_freq (int): Min word frequency.
           padding (str): Padding token.
           unknown (str): Unknown token.
           dataset_type (str): Dataset type(train, dev, test).
               Different types of datasets may be preprocessed differently.

        Returns:
           DataFrame: Preprocessed dataset.
       """
        if "label" in dataset.columns.values:
            if not self._label_is_float and isinstance(self._label_map, Dict):
                dataset["label"] = dataset["label"].map(self.label_to_idx)
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["sentence1"] = self.tokenize_progress(dataset, dataset_type, "sentence1")
            dataset["sentence2"] = self.tokenize_progress(dataset, dataset_type, "sentence2")

            if dataset_type == "train":
                self._label_nums = dataset["label"].value_counts().shape[0]
                self._vocab = Vocabulary.from_dataset(dataset, field_name=["sentence1", "sentence2"], max_size=max_size,
                                                      min_freq=min_freq, padding=padding, unknown=unknown)
            dataset["input1_ids"] = self._vocab.word_to_idx(dataset["sentence1"])
            dataset["input2_ids"] = self._vocab.word_to_idx(dataset["sentence2"])
            dataset.drop("sentence1", axis=1, inplace=True)
            dataset.drop("sentence2", axis=1, inplace=True)
            dataset["input1_length"] = self.get_length_progress(dataset, dataset_type, "input1_ids")
            dataset["input2_length"] = self.get_length_progress(dataset, dataset_type, "input2_ids")
            if not self._buckets:
                if isinstance(self._max_length, int):
                    max_length1 = self._max_length
                else:
                    max_length1 = dataset["input1_length"].max()
                if isinstance(self._max_pair_length, int):
                    max_length2 = self._max_pair_length
                else:
                    max_length2 = dataset["input2_length"].max()
                pad1 = Pad(max_length=max_length1, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
                pad2 = Pad(max_length=max_length2, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
                dataset["input1_ids"] = self.padding_progress(dataset, dataset_type, field="input1_ids",
                                                              pad_function=pad1)
                dataset["input2_ids"] = self.padding_progress(dataset, dataset_type, field="input2_ids",
                                                              pad_function=pad2)
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)
                dataset["input1_ids"] = self.padding_progress(dataset, dataset_type, field="input1_ids",
                                                              pad_function=pad)
                dataset["input2_ids"] = self.padding_progress(dataset, dataset_type, field="input2_ids",
                                                              pad_function=pad)

                dataset[["input1_ids", "input2_ids"]] = self.padding_same_progress(dataset, dataset_type,
                                                                                   ["input1_ids", "input2_ids"])
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input1_ids")

        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())

            dataset_tokenized = DataFrame(
                self.tokenize_progress(dataset, dataset_type, field=["sentence1", "sentence2"]))
            dataset.drop("sentence1", axis=1, inplace=True)
            dataset.drop("sentence2", axis=1, inplace=True)

            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                dataset_tokenized.columns = self._pretrained_model_inputs
                self._max_length = dataset_tokenized["length"].max()
                dataset_tokenized = DataFrame(
                    self.padding_progress(dataset_tokenized, dataset_type, pad_function=self._tokenizer.pad))
            dataset_tokenized.columns = self._pretrained_model_inputs
            if isinstance(self._buckets, List):
                dataset_tokenized["padding_length"] = self.get_length_progress(dataset_tokenized, dataset_type,
                                                                               "input_ids")
            if "label" in dataset.columns.values:
                dataset_tokenized["label"] = dataset["label"]
            dataset = dataset_tokenized
            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                self._pretrained_model_inputs.remove("length")
        return dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]],
                     process_function: callable = None) -> List[str]:
        """
        Write PairCLSDataset to .mindrecord file.

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
                "input1_ids": {"type": "int32", "shape": [-1]},
                "input1_length": {"type": "int32", "shape": [-1]},
                "input2_ids": {"type": "int32", "shape": [-1]},
                "input2_length": {"type": "int32", "shape": [-1]}}
        else:
            data_schema = {}
            for i in self._pretrained_model_inputs:
                data_schema[i] = {"type": "int32", "shape": [-1]}

        if callable(process_function):
            colmun_names = dataset.iterrows()
            i, row = next(colmun_names)
            row = process_function(row)
            if "label" in row.keys():
                if not self._label_is_float:
                    data_schema["label"] = {"type": "int32", "shape": [-1]}
                else:
                    data_schema["label"] = {"type": "float32", "shape": [-1]}
        else:
            if "label" in dataset.columns.values:
                if not self._label_is_float:
                    data_schema["label"] = {"type": "int32", "shape": [-1]}
                else:
                    data_schema["label"] = {"type": "float32", "shape": [-1]}

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
                sample = {"input1_ids": np.array(row["input1_ids"], dtype=np.int32),
                          "input1_length": np.array(row["input1_length"], dtype=np.int32),
                          "input2_ids": np.array(row["input2_ids"], dtype=np.int32),
                          "input2_length": np.array(row["input2_length"], dtype=np.int32)}
            else:
                sample = {}
                for i in self._pretrained_model_inputs:
                    sample[i] = np.array(row[i], dtype=np.int32)

            if callable(process_function):
                if "label" in row.keys():
                    if not self._label_is_float:
                        sample["label"] = np.array(row["label"], dtype=np.int32)
                    else:
                        sample["label"] = np.array(row["label"], dtype=np.float32)
            else:
                if "label" in dataset.columns.values:
                    if not self._label_is_float:
                        sample["label"] = np.array(row["label"], dtype=np.int32)
                    else:
                        sample["label"] = np.array(row["label"], dtype=np.float32)

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

    @property
    def label_nums(self) -> int:
        """
        Return label_nums.
        """
        return self._label_nums

    @label_nums.setter
    def label_nums(self, nums: int):
        """
        Need to be assigned.

        Args:
            nums (str): The number of label.
        """
        self._label_nums = nums


class GenerateBaseDataset(Dataset):
    """
    A base class of text generation.
   """

    def __init__(self, **kwargs):
        super(GenerateBaseDataset, self).__init__(**kwargs)
        if self._stream:
            if (not isinstance(self._max_length, int) or not isinstance(self._max_pair_length, int)) and not isinstance(
                    self._buckets, List):
                raise TypeError(
                    "`max_length`, `max_pair_length` or `buckets` should be assigned when `stream` is `True`.")
        if bool(self._truncation_strategy) and not (
                isinstance(self._max_length, int) or isinstance(self._max_pair_length, int)) and not isinstance(
                    self._buckets, List):
            raise TypeError(
                "`truncation_strategy` need be `False` when `max_length` or `max_pair_length` is not assigned.")

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        """
        Preprocess dataset by data stream.

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str):  Dataset type(train, dev, test).
                Different types of datasets may be processed differently.

        Returns:
            callable: A preprocess function.
        """
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset['document'] = self.tokenize_progress(dataset, dataset_type, 'document')
            dataset['summary'] = self.tokenize_progress(dataset, dataset_type, 'summary')

            if dataset_type == 'train':
                self._vocab = Vocabulary.from_dataset(dataset, field_name=['document', 'summary'], max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)

            if not self._buckets:
                pad1 = Pad(max_length=self._max_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)
                pad2 = Pad(max_length=self._max_pair_length, pad_val=self._vocab.padding_idx,
                           truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input_ids": [self._vocab[i] for i in row["document"]],
                            "output_ids": [self._vocab[i] for i in row["summary"]]}
                    data["input_length"] = len(data["input_ids"])
                    data["output_length"] = len(data["output_ids"])
                    data["input_ids"] = pad1(data["input_ids"])
                    data["output_ids"] = pad2(data["output_ids"])
                    return data
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)

                def token_to_idx(row):
                    data = {"input_ids": [self._vocab[i] for i in row["document"]],
                            "output_ids": [self._vocab[i] for i in row["summary"]]}
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
                    model_inputs = self._tokenizer(row["document"], truncation=self._truncation_strategy,
                                                   padding="max_length", max_length=self._max_length)
                    with self._tokenizer.as_target_tokenizer():
                        label = self._tokenizer(row["summary"], truncation=self._truncation_strategy,
                                                padding="max_length", max_length=self._max_pair_length)
                    model_inputs["labels"] = label["input_ids"]
                    return model_inputs
            else:
                def token_to_idx(row):
                    document_length = len(self._tokenizer.tokenize(row["document"], add_special_tokens=True))
                    summary_length = len(self._tokenizer.tokenize(row["summary"], add_special_tokens=True))
                    d_i = 0
                    for d_i in self._buckets:
                        if d_i >= document_length:
                            break
                    s_i = 0
                    for s_i in self._buckets:
                        if s_i >= summary_length:
                            break
                    i = d_i if d_i > s_i else s_i
                    model_inputs = self._tokenizer(row["document"], truncation=self._truncation_strategy,
                                                   padding="max_length", max_length=i)

                    with self._tokenizer.as_target_tokenizer():
                        label = self._tokenizer(row["summary"], truncation=self._truncation_strategy,
                                                padding="max_length", max_length=i)
                    model_inputs["labels"] = label["input_ids"]
                    model_inputs["padding_length"] = len(model_inputs["input_ids"])
                    return model_inputs
        return token_to_idx

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str,
                 unknown: str, dataset_type: str) -> DataFrame:
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["document"] = self.tokenize_progress(dataset, dataset_type, "document")
            dataset["summary"] = self.tokenize_progress(dataset, dataset_type, "summary")

            if dataset_type == 'train':
                self._vocab = Vocabulary.from_dataset(dataset, field_name=['document', 'summary'], max_size=max_size,
                                                      min_freq=min_freq,
                                                      padding=padding, unknown=unknown)
            dataset["input_ids"] = self._vocab.word_to_idx(dataset["document"])
            dataset["output_ids"] = self._vocab.word_to_idx(dataset["summary"])
            dataset.drop("document", axis=1, inplace=True)
            dataset.drop("summary", axis=1, inplace=True)
            dataset["input_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
            dataset["output_length"] = self.get_length_progress(dataset, dataset_type, "output_ids")
            if not self._buckets:
                if isinstance(self._max_length, int):
                    max_length1 = self._max_length
                else:
                    max_length1 = dataset["input_length"].max()
                if isinstance(self._max_pair_length, int):
                    max_length2 = self._max_pair_length
                else:
                    max_length2 = dataset["output_length"].max()
                pad1 = Pad(max_length=max_length1, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
                pad2 = Pad(max_length=max_length2, pad_val=self._vocab.padding_idx, truncate=self._truncation_strategy)
                dataset["input_ids"] = self.padding_progress(dataset, dataset_type, field="input_ids",
                                                             pad_function=pad1)
                dataset["output_ids"] = self.padding_progress(dataset, dataset_type, field="output_ids",
                                                              pad_function=pad2)
            else:
                pad = Pad(pad_val=self._vocab.padding_idx, buckets=self._buckets, truncate=self._truncation_strategy)
                dataset["input_ids"] = self.padding_progress(dataset, dataset_type, field="input_ids",
                                                             pad_function=pad)
                dataset["output_ids"] = self.padding_progress(dataset, dataset_type, field="output_ids",
                                                              pad_function=pad)
                dataset[["input_ids", "output_ids"]] = self.padding_same_progress(dataset, dataset_type,
                                                                                  ["input_ids", "output_ids"])
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
        else:
            self._pretrained_model_inputs_document = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_length, int)).data.keys())
            self._pretrained_model_inputs_summary = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List) and not isinstance(
                    self._max_pair_length, int)).data.keys())
            document = DataFrame(self.tokenize_progress(dataset, dataset_type, field="document"))
            dataset.drop("document", axis=1, inplace=True)

            temp_pair_length = self._max_pair_length
            temp_length = self._max_length
            self._max_length = self._max_pair_length

            summary = DataFrame(self.tokenize_progress(dataset, dataset_type, field="summary"))
            dataset.drop("summary", axis=1, inplace=True)

            self._max_pair_length = temp_pair_length
            self._max_length = temp_length

            def document_list_split(row):
                data = row["document"]
                return tuple(data)

            tqdm.pandas(desc=f"{self._name} {dataset_type} dataset processing.")
            document = document.progress_apply(document_list_split, axis=1, result_type="expand")

            def summary_list_split(row):
                data = row["summary"]
                return tuple(data)

            tqdm.pandas(desc=f"{self._name} {dataset_type} dataset processing.")
            summary = summary.progress_apply(summary_list_split, axis=1, result_type="expand")
            if not isinstance(self._buckets, List) and not isinstance(self._max_length, int):
                document.columns = self._pretrained_model_inputs_document
                self._max_length = document["length"].max()
                document = DataFrame(
                    self.padding_progress(document, dataset_type, pad_function=self._tokenizer.pad))
                document.columns = self._pretrained_model_inputs_document
                document.drop("length", axis=1, inplace=True)
                self._pretrained_model_inputs_document.remove("length")
            else:
                document.columns = self._pretrained_model_inputs_document

            if not isinstance(self._buckets, List) and not isinstance(self._max_pair_length, int):
                summary.columns = self._pretrained_model_inputs_summary
                self._max_pair_length = summary["length"].max()
                temp_pair_length = self._max_pair_length
                temp_length = self._max_length
                self._max_length = self._max_pair_length
                summary = DataFrame(
                    self.padding_progress(summary, dataset_type, pad_function=self._tokenizer.pad))
                self._max_pair_length = temp_pair_length
                self._max_length = temp_length
                summary.columns = self._pretrained_model_inputs_summary
                summary.drop("length", axis=1, inplace=True)
                self._pretrained_model_inputs_summary.remove("length")
            else:
                summary.columns = self._pretrained_model_inputs_summary

            dataset[document.columns] = document
            dataset["labels"] = summary["input_ids"]
            del document
            del summary
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
                    dataset_group.columns = self._pretrained_model_inputs_summary
                    dataset['labels'][dataset_group.index] = dataset_group['input_ids']
                dataset["padding_length"] = self.get_length_progress(dataset, dataset_type, "input_ids")
            self._pretrained_model_inputs = self._pretrained_model_inputs_document
        return dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]],
                     process_function: callable = None) -> List[str]:
        """
        Write GenerateBaseDataset to .mindrecord file.

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
        if not self._process4transformer:
            if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                data_schema = {
                    'input_ids': {'type': 'int32', 'shape': [-1]},
                    'input_length': {'type': 'int32', 'shape': [-1]}}
            else:
                data_schema = {}
                for i in self._pretrained_model_inputs:
                    data_schema[i] = {'type': 'int32', 'shape': [-1]}
        else:
            data_schema = {}

        if callable(process_function):
            colmun_names = dataset.iterrows()
            i, row = next(colmun_names)
            row = process_function(row)
            if self._process4transformer:
                for field in row.keys():
                    data_schema[field] = {"type": "int64", "shape": [-1]}
                if isinstance(self._buckets, List):
                    del data_schema['padding_length']
            else:
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
            if not self._process4transformer:
                if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
                    sample = {'input_ids': np.array(row["input_ids"], dtype=np.int64),
                              'input_length': np.array(row["input_length"], dtype=np.int64)}
                else:
                    sample = {}
                    for i in self._pretrained_model_inputs:
                        sample[i] = np.array(row[i], dtype=np.int64)
            else:
                sample = {}

            if callable(process_function):
                if self._process4transformer:
                    for field, _ in data_schema.items():
                        sample[field] = np.array(row[field], dtype=np.int64)

                else:
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
