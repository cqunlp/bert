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
     Tokenizer utils.
"""
import unicodedata
from enum import Enum

SPIECE_UNDERLINE = "▁"


def _is_control(char):
    """
    Check whether `char` is a control character.
    """
    if char in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """
    Check whether `char` is a punctuation character.
    """
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_whitespace(char):
    """
    Check whether `char` is a whitespace character.
    """
    if char in (" ", "\t", "\n", "\r"):
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_end_of_word(text):
    """
    Check whether the last character in `text` is a punctuation, whitespace, control character.
    """
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """
    Check whether the first character in `text` is a punctuation, whitespace, control character.
    """
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


class PaddingStrategy(Enum):
    """
    Possible values for the `padding` argument in `PreTrainedTokenizer.__call__`.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TruncationStrategy(Enum):
    """
    Possible values for the `truncation` argument in `PreTrainedTokenizer.__call__`.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


def get_padding_truncation_strategies(padding=False, truncation=False, max_length=None, kwargs=None):
    """
    Get pad strategies and truncation strategies according to `padding`, 'truncation' and `max_length`.
    """
    if truncation is True:
        truncation_strategy = TruncationStrategy.LONGEST_FIRST
    elif isinstance(truncation, str):
        truncation_strategy = TruncationStrategy(truncation)
    else:
        truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

    padding_strategy = PaddingStrategy.DO_NOT_PAD
    if padding is False:
        if max_length is None:
            padding_strategy = PaddingStrategy.LONGEST
        else:
            padding_strategy = PaddingStrategy.MAX_LENGTH
    elif padding is not False:
        if padding is True:
            padding_strategy = PaddingStrategy.LONGEST
        elif isinstance(padding, str):
            padding_strategy = PaddingStrategy(padding)

    return padding_strategy, truncation_strategy, max_length, kwargs
