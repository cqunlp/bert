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
    SQuAD dataset
"""
import json
from typing import Union, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
from ..base_dataset import Dataset
from ...common import Pad, Vocabulary
from .. import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.DATASET)
class SquadDataset(Dataset):
    """
    SQuAD dataset load.

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
        >>> squad = SquadDataset(tokenizer='studio-ousia/luke-large')
          # squad = SquadDataset(tokenizer='studio-ousia/luke-large')
        >>> dataset = squad()
    """

    def __init__(self, paths: Union[str, Dict[str, str]] = None,
                 tokenizer: Union[str] = 'studio-ousia/luke-large', lang: str = 'en', max_size: int = None,
                 min_freq: int = None,
                 padding: str = '<pad>', unknown: str = '<unk>',
                 buckets: List[int] = None):
        super(SquadDataset, self).__init__(sep='\t', name='squad')
        self._paths = paths
        self._tokenize = tokenizer
        self._lang = lang
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

    def _load(self, path: str) -> DataFrame:
        """
        Load dataset from SQuAD file.

        Args:
            path (str): Dataset file path.

        Returns:
            DataFrame: Dataset file will be read as a DataFrame.
        """
        with open(path, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        dataset = pd.DataFrame(columns=('id',
                                        'title'
                                        'context',
                                        'question_text',
                                        'start_position',
                                        'end_position',
                                        'orig_answer_text',
                                        'is_impossible'))
        all_count = 0
        real_count = 0
        is_impossible = False
        for entry in input_data:
            title = entry.get("title", "").strip()
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    all_count += 1
                    if qa["answers"][0]["answer_start"] != -1:
                        dataset = dataset.append({'id': qa["id"],
                                                  'title': title,
                                                  'context': paragraph_text,
                                                  'question_text': qa["question"].strip(),
                                                  'start_position': qa["answers"][0]["answer_start"],
                                                  'end_position': qa["answers"][0]["answer_start"] + len(
                                                      qa["answers"][0]["text"]) - 1,
                                                  'orig_answer_text': qa["answers"][0]["text"],
                                                  'is_impossible': is_impossible}, ignore_index=True)

                        real_count += 1
        return dataset

    def _stream_process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str, unknown: str,
                        dataset_type: str) -> callable:
        raise NotImplementedError(f"{self.__class__} cannot be preprocessed by data stream.")

    def _process(self, dataset: DataFrame, max_size: int, min_freq: int, padding: str,
                 unknown: str, dataset_type: str) -> DataFrame:
        """
        LUKE Model preprocess

        Args:
            dataset (DataFrame): DataFrame need to preprocess.
            max_size (int): Vocab max size.
            min_freq (int): Min word frequency.
            padding (str): Padding token.
            unknown (str): Unknown token.
            dataset_type (str): Dataset type(train, dev, test).
                Different types of datasets may be preprocessed differently.

        Returns:
            Dict[str, MindDataset]: A MindDataset dictionary.
        """
        # Whether using a pretrained model tokenizer.
        if not isinstance(self._tokenizer, PreTrainedTokenizerBase):
            dataset["context"] = self.tokenize_progress(dataset, dataset_type, 'sentence1')
            dataset["question_text"] = self.tokenize_progress(dataset, dataset_type, 'sentence2')
            if dataset_type != 'test':
                dataset['start_position'] = dataset['start_position']
                dataset['end_position'] = dataset['end_position']

            if dataset_type == 'train':
                self._label_nums = dataset['label'].value_counts().shape[0]
                self._vocab = Vocabulary.from_dataset(dataset, field_name=["context", "question_text"],
                                                      max_size=max_size, min_freq=min_freq,
                                                      padding=padding, unknown=unknown)
            dataset['input_ids_q'] = self._vocab.word_to_idx(dataset['question_text'])
            dataset['input_ids_c'] = self._vocab.word_to_idx(dataset['context'])
            dataset.drop('input_ids_q', axis=1, inplace=True)
            dataset.drop('input_ids_c', axis=1, inplace=True)
            dataset['input_ids_q'] = self.get_length_progress(dataset, dataset_type, 'input_ids_q')
            dataset['input_ids_c'] = self.get_length_progress(dataset, dataset_type, 'input_ids_c')
            if not self._buckets:
                if isinstance(self._max_length, int):
                    max_length1 = self._max_length
                else:
                    max_length1 = dataset['input_ids_q'].max()
                if isinstance(self._max_pair_length, int):
                    max_length2 = self._max_pair_length
                else:
                    max_length2 = dataset['input_ids_c'].max()
                pad1 = Pad(max_length1, self._vocab.padding_idx)
                pad2 = Pad(max_length2, self._vocab.padding_idx)
                dataset['input_ids_q'] = self.padding_progress(dataset, dataset_type, field='input_ids_q',
                                                               pad_function=pad1)
                dataset['input_ids_c'] = self.padding_progress(dataset, dataset_type, field='input_ids_c',
                                                               pad_function=pad2)
            else:
                pad = Pad(self._vocab.padding_idx, buckets=self._buckets)
                dataset['input_ids_q'] = self.padding_progress(dataset, dataset_type, field='input_ids_q',
                                                               pad_function=pad)
                dataset['input_ids_c'] = self.padding_progress(dataset, dataset_type, field='input_ids_c',
                                                               pad_function=pad)

                dataset[['input_ids_q', 'input_ids_c']] = self.padding_same_progress(dataset, dataset_type,
                                                                                     ['input_ids_q', 'input_ids_c'])
                dataset['padding_length_q'] = self.get_length_progress(dataset, dataset_type, 'input_ids_q')
                dataset['padding_length_c'] = self.get_length_progress(dataset, dataset_type, 'input_ids_c')

        else:
            self._pretrained_model_inputs = list(
                self._tokenizer("", return_length=not isinstance(self._buckets, List)).data.keys())
            if dataset_type != 'test':
                dataset['start_position'] = dataset['start_position']
                dataset['end_position'] = dataset['end_position']
            dataset_tokenized = DataFrame(
                self.tokenize_progress(dataset, dataset_type, field=["context", "question_text"]))
            dataset.drop("context", axis=1, inplace=True)
            dataset.drop("question_text", axis=1, inplace=True)

            if not isinstance(self._buckets, List):
                dataset_tokenized.columns = self._pretrained_model_inputs
                if isinstance(self._max_length, int):
                    self._buckets = self._max_length
                else:
                    self._buckets = dataset_tokenized['length'].max()
                dataset_tokenized = DataFrame(
                    self.padding_progress(dataset_tokenized, dataset_type, pad_function=self._tokenizer.pad))
            dataset_tokenized.columns = self._pretrained_model_inputs
            if isinstance(self._buckets, List):
                dataset_tokenized['padding_length'] = self.get_length_progress(dataset_tokenized, dataset_type,
                                                                               'input_ids')
            dataset = dataset_tokenized
            if not isinstance(self._buckets, List):
                self._pretrained_model_inputs.remove("length")
        return dataset

    def _write_to_mr(self, dataset: DataFrame, file_path: Union[str, Dict[int, str]],
                     process_function: callable = None) -> List[str]:
        """
        Write RCDataset to .mindrecord file.

        Args:
            dataset (DataFrame): Tokenizer function.
            file_path (str): Path of mindrecord file.
            process_function (callable): A function is used to preprocess data.

        Returns:
            List[str]: Dataset field
        """
        writer = FileWriter(file_name=file_path, shard_num=1)
        data_schema = {
            "unique_id": {"type": "int32", "shape": [-1]},
            "word_ids": {"type": "int32", "shape": [-1]},
            "word_segment_ids": {"type": "int32", "shape": [-1]},
            "word_attention_mask": {"type": "int32", "shape": [-1]},
            "entity_ids": {"type": "int32", "shape": [-1]},
            "entity_position_ids": {"type": "int32", "shape": [-1]},
            "entity_segment_ids": {"type": "int32", "shape": [-1]},
            "entity_attention_mask": {"type": "int32", "shape": [-1]},
        }

        if ("start_position" in dataset.columns.values) and ("end_position" in dataset.columns.values):
            data_schema['start_position'] = {'type': 'int32', 'shape': [-1]}
            data_schema['end_position'] = {'type': 'int32', 'shape': [-1]}
        writer.add_schema(data_schema, self._name)
        data = []
        vocab_bar = tqdm(dataset.iterrows(), total=len(dataset))
        for index, row in vocab_bar:
            sample = {"unique_id": np.array(row["unique_id"], dtype=np.int32),
                      "word_ids": np.array(row["word_ids"], dtype=np.int32),
                      "word_segment_ids": np.array(row["word_segment_ids"], dtype=np.int32),
                      "word_attention_mask": np.array(row["word_attention_mask"], dtype=np.int32),
                      "entity_ids": np.array(row["entity_ids"], dtype=np.int32),
                      "entity_position_ids": np.array(row["entity_position_ids"], dtype=np.int32),
                      "entity_segment_ids": np.array(row["entity_segment_ids"], dtype=np.int32),
                      "entity_attention_mask": np.array(row["entity_attention_mask"], dtype=np.int32),
                      }
            if ("start_position" in dataset.columns.values) and ("end_position" in dataset.columns.values):
                sample['start_position'] = np.array(row['start_position'], dtype=np.int32)
                sample['end_position'] = np.array(row['end_position'], dtype=np.int32)
            data.append(sample)
            if index % 10 == 0:
                writer.write_raw_data(data)
                data = []
            vocab_bar.set_description("Writing data to .mindrecord file")
        if data:
            writer.write_raw_data(data)
        writer.commit()
        return list(data_schema.keys())
