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
"""vocabulary class"""
from typing import List, Union, Dict, Optional
import collections
from collections import Counter

import pandas as pd
from tqdm import tqdm


class Vocabulary:
    """
    Convert word to index.

    Args:
        max_size (int, Optional): Vocab max size, default None.
        min_freq (int, Optional): Min word frequency, default None.
        padding (str): Padding token, default `<pad>`.
        unknown (str): Unknown token, default `<unk>`.

    Examples:
        >>> vocab = Vocabulary()
        >>> word_list = "this is a word list".split()
        >>> vocab.update(word_list)
        >>> vocab["word"] # tokens to int
        >>> vocab.to_word(5) # int to tokens
        >>> vocab.build_vocab() # build vocabulary
    """

    def __init__(self, max_size: Optional[int] = None, min_freq: Optional[int] = None, padding: str = '<pad>',
                 unknown: str = '<unk>'):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_count = Counter()
        self.unknown = unknown
        self.padding = padding
        self.padding_idx = None
        self.unknown_idx = None
        self._word2idx = None
        self._idx2word = None

    def add(self, word: str):
        """
        Increase the frequency of a word.

        Args:
            word (str): A word.
        """
        self.word_count[word] += 1

    def update(self, word_list: List[str]):
        """
        Increase the frequency of multiple words.

        Args:
            word_list (List[str]): A word list.
        """
        self.word_count.update(word_list)

    def build_vocab(self):
        """
        Build a dictionary based on word frequency.
        """
        if not self._word2idx:
            self._word2idx = {}
            if self.padding != '':
                self._word2idx[self.padding] = len(self._word2idx)
            if (self.unknown != '') and (self.unknown != self.padding):
                self._word2idx[self.unknown] = len(self._word2idx)

        max_size = min(self.max_size, len(self.word_count)) if self.max_size else None
        words = self.word_count.most_common(max_size)
        if isinstance(self.min_freq, int):
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if isinstance(self._word2idx, Dict):
            words = filter(lambda kv: kv[0] not in self._word2idx, words)
        start_idx = len(self._word2idx)
        self._word2idx.update({w: i + start_idx for i, (w, _) in enumerate(tqdm(words))})
        self._idx2word = {i: w for w, i in tqdm(self._word2idx.items())}
        self.padding_idx = self[self.padding]
        self.unknown_idx = self[self.unknown]

    def to_word(self, idx: int) -> str:
        """
        Given a number convert to the corresponding token.

        Args:
            idx (int): A index of token.

        Returns:
            str: Token.
        """
        return self._idx2word[idx]

    def __getitem__(self, word: str) -> int:
        """
        Return token index.

        Args:
            word (str): Token.

        Returns:
            int: A index of token.
        """
        idx = self._word2idx.get(word, self.unknown_idx)
        if not idx and self.unknown_idx == '':
            raise ValueError(f"word `{word}` not in vocabulary")
        return idx

    def __len__(self):
        return len(self._word2idx)

    def word_to_idx(self, word: pd.Series) -> pd.Series:
        """
        Convert tokens to index.

        Args:
            word (Series): Series needed to convert to index.

        Returns:
            Series: Converted Series.
        """
        tqdm.pandas(desc=f"Convert tokens to index.")
        index = word.progress_apply(lambda n: [self[i] for i in n])
        return index

    def idx_to_word(self, index: pd.Series) -> pd.Series:
        """
        Convert index to tokens.

        Args:
            index (Series): Series needed to convert to tokens.

        Returns:
            Series: Converted dataset.
        """
        tqdm.pandas(desc=f"Convert index(`{index.name}` field) to tokens.")
        word = index.progress_apply(lambda n: [self.to_word(i) for i in n])
        return word

    @property
    def word2idx(self):
        return self._word2idx

    @property
    def idx2word(self):
        return self._idx2word

    @word2idx.setter
    def word2idx(self, value):
        self._word2idx = value

    @idx2word.setter
    def idx2word(self, value):
        self._idx2word = value

    @staticmethod
    def from_dataset(dataset: pd.DataFrame, field_name: Union[str, List[str]], max_size: Optional[int] = None,
                     min_freq: Optional[int] = None, padding: str = '<pad>', unknown: str = '<unk>'):
        """
        Build a Vocabulary from a dataset.

        Args:
            dataset (DataFrame): Dataset.
            field_name (Union[str, List[str]]): Which field of dataset need to be built.
            max_size (int, Optional): Vocabulary max size, default None.
            min_freq (int, Optional): Min word frequency, default None.
            padding (str): Padding token, default `<pad>`.
            unknown (str): Unknown token, default `<unk>`.

        Returns:
            Vocabulary: Vocabulary built from a dataset.
        """
        vocab = Vocabulary(max_size=max_size, min_freq=min_freq, padding=padding, unknown=unknown)
        field_name = [field_name] if isinstance(field_name, str) else field_name
        if isinstance(field_name, (str, List)):
            vocab_bar = tqdm(dataset[field_name].iterrows(), total=len(dataset))
            for _, row in vocab_bar:
                for sent in field_name:
                    vocab.update(row[sent])
                vocab_bar.set_description("Build Vocabulary")
            vocab.build_vocab()
        return vocab

    @staticmethod
    def from_file(path: str):
        """
        Build a Vocabulary from a file.

        Args:
            path (str): Vocab file path

        Returns:
            Vocabulary: Vocabulary built from a vocab file.
        """
        vocab = Vocabulary()
        vocab_dict = load_vocab_file(path)
        vocab.word2idx = vocab_dict
        vocab.idx2word = {i: w for w, i in tqdm(vocab_dict.items())}
        return vocab


def load_vocab_file(vocab_file: str) -> dict:
    """
    Loads a vocabulary file and turns into a {token:id} dictionary.
    """
    vocab_dict = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf-8') as vocab:
        while True:
            token = vocab.readline()
            if not token:
                break
            token = token.strip()
            vocab_dict[token] = index
            index += 1
    return dict(vocab_dict)
