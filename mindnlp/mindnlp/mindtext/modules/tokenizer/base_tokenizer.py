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
    Base Tokenization classes.
"""
import itertools
import re
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Optional, List, Union, Dict, Tuple

from .utils import PaddingStrategy, TruncationStrategy, get_padding_truncation_strategies, _is_start_of_word, \
    _is_end_of_word


@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    A added token class, used to create a special token cannot be splited.
    """
    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __str__(self):
        return self.content

    def __getstate__(self):
        return self.__dict__


class SpecialTokens:
    """
    A special tokens class, used to store common special tokens such as `[CLS]`, `[SEP]`.
    """
    SPECIAL_TOKENS_ATTRIBUTES = ["bos_token",
                                 "eos_token",
                                 "unk_token",
                                 "sep_token",
                                 "pad_token",
                                 "cls_token",
                                 "mask_token"]

    def __init__(self, **kwargs):
        self.bos_token = kwargs.pop("bos_token", None)
        self.eos_token = kwargs.pop("eos_token", None)
        self.unk_token = kwargs.pop("unk_token", None)
        self.sep_token = kwargs.pop("sep_token", None)
        self.pad_token = kwargs.pop("pad_token", None)
        self.cls_token = kwargs.pop("cls_token", None)
        self.mask_token = kwargs.pop("mask_token", None)
        self.pad_token_type_id = 0

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken]]:
        """
        All special tokens dictionary.

        Returns:
            Dict[str, Union[str, AddedToken]]: Return a dict contains all special tokens. For example:
                `{'bos_token':'[CLS]','sep_token':'[SEP]'}` or
                `{'bos_token':AddedToken('[CLS]'),'sep_token':AddedToken('[SEP]')}`.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        All special tokens list.

        Returns:
            List[Union[str, AddedToken]]: Return a list contains all special tokens. For example:
                `['[CLS]','[SEP]']` or
                `[AddedToken('[CLS]'),AddedToken('[SEP]')]`.
        """
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
            all_toks = all_toks + (list(attr_value) if isinstance(attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_tokens(self) -> List[str]:
        """
        All special tokens(casted to string) list.

        Returns:
            List[str]: Return a list contains all special tokens(casted to string). For example:
                `['[CLS]','[SEP]']`.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks


class PreTrainedTokenizer(SpecialTokens):
    """
    Base class for tokenizer.

    Args:
        model_input_names (List[str]): A pretrained model input name,
            default ["input_ids", "token_type_ids", "attention_mask"].
        padding_side (str): Padding left or right,default `right`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_input_names = kwargs.pop("model_input_names", ["input_ids", "token_type_ids", "attention_mask"])
        self.padding_side = kwargs.pop("padding_side", "right")
        self.unique_no_split_tokens = [str(i) for i in self.all_special_tokens_extended]

    def __call__(self,
                 text: str,
                 text_pair: Optional[str] = None,
                 add_special_tokens: bool = True,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False,
                 max_length: Optional[int] = None,
                 return_length: bool = False,
                 return_token_type_ids: bool = True,
                 **kwargs) -> Dict[str, List[int]]:
        """
        Tokenize and prepare for the model one or pair of sequences.

        Args:
            text (str): The sequence will be encoded.
            text_pair (str, Optional): The sequence will be encoded,default None.

        Returns:
            Dict[str, List[int]]: Return a dictionary contains tokenized index sequence.
                The key of the dictionary consists of `model_input_names`. For example:
                    {'input_ids':'...',
                     'token_type_ids':'...',
                     'attention_mask':'...',}.
        """
        return self.encode(text=text,
                           text_pair=text_pair,
                           add_special_tokens=add_special_tokens,
                           padding=padding,
                           truncation=truncation,
                           max_length=max_length,
                           return_length=return_length,
                           return_token_type_ids=return_token_type_ids,
                           **kwargs)

    def encode(self,
               text: str,
               text_pair: Optional[str] = None,
               add_special_tokens: bool = True,
               padding: Union[bool, str] = False,
               truncation: Union[bool, str] = False,
               max_length: Optional[int] = None,
               return_length: bool = False,
               return_token_type_ids: bool = True,
               **kwargs) -> Dict[str, List[int]]:
        """
        The main method to tokenize and prepare for the model one or pair of sequences.

        Args:
            text (str): The sequence will be encoded.
            text_pair (str, Optional): The sequence will be encoded,default None.

        Returns:
            Dict[str, List[int]]: Return a dictionary contains tokenized index sequence.
                The key of the dictionary consists of `model_input_names`. For example:
                    {'input_ids':'...',
                     'token_type_ids':'...',
                     'attention_mask':'...',}.
        """

        def get_input_ids(text_input):
            tokens = self.tokenize(text_input)
            return self.convert_tokens_to_ids(tokens)

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_inputs(ids=first_ids,
                                   ids_pair=second_ids,
                                   add_special_tokens=add_special_tokens,
                                   padding=padding,
                                   truncation=truncation,
                                   max_length=max_length,
                                   return_length=return_length,
                                   return_token_type_ids=return_token_type_ids,
                                   **kwargs)

    def prepare_inputs(self,
                       ids: List[int],
                       ids_pair: Optional[List[int]] = None,
                       add_special_tokens: bool = True,
                       padding: Union[bool, str] = False,
                       truncation: Union[bool, str] = False,
                       max_length: Optional[int] = None,
                       return_length: bool = False,
                       return_token_type_ids: bool = True,
                       **kwargs
                       ) -> Dict[str, List[int]]:
        """
        The method to prepare inputs for model.(pad, truncate etc.).

        Args:
            ids (List[int]): The index of tokenized sequences.
            ids_pair (List[int], Optional): The index of tokenized sequences,default None.
        """
        padding_strategy, truncation_strategy, _, _ = get_padding_truncation_strategies(padding=padding,
                                                                                        truncation=truncation,
                                                                                        kwargs=kwargs)
        pair = bool(ids_pair is not None)
        len_ids = len(ids)
        len_ids_pair = len(ids_pair) if pair else 0

        encoded_inputs = {}

        total_len = len_ids + len_ids_pair + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, ids_pair, _ = self.truncate_sequences(ids,
                                                       ids_pair=ids_pair,
                                                       num_tokens_to_remove=total_len - max_length,
                                                       truncation_strategy=truncation_strategy)

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, ids_pair)
            if return_token_type_ids:
                token_type_ids = self.create_token_type_ids_from_sequences(ids, ids_pair)
        else:
            sequence = ids + ids_pair if pair else ids
            if return_token_type_ids:
                token_type_ids = [0] * len(ids) + ([0] * len(ids_pair) if pair else [])

        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            encoded_inputs = self.pad(encoded_inputs,
                                      padding=padding_strategy.value,
                                      max_length=max_length)

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        return encoded_inputs

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[None, int, List[int]]:
        """
        Converts a token (or a sequence of tokens) in a integer id (or a sequence of ids).
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def convert_ids_to_tokens(self, ids: List[int]):
        """
        Converts a id (or a sequence of ids) in a token (or a sequence of tokens).
        """
        tokens = []
        for index in ids:
            index = int(index)
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def pad(self,
            encoded_inputs: Dict[str, List[int]],
            padding: Union[bool, str] = True,
            max_length: Optional[int] = None) -> Dict[str, List[int]]:
        """
        Pad a sequence of ids to `max_length`
        """
        padding_strategy, _, max_length, _ = get_padding_truncation_strategies(padding=padding, max_length=max_length)
        encoded_inputs = self._pad(encoded_inputs,
                                   max_length=max_length,
                                   padding_strategy=padding_strategy)
        return encoded_inputs

    def tokenize(self, text: str) -> List[str]:
        """
        Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.
        """
        all_special_tokens_extended = dict((str(t), t)
                                           for t in self.all_special_tokens_extended if isinstance(t, AddedToken))

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            escaped_special_toks = [re.escape(s_tok) for s_tok in self.all_special_tokens]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        if (i < len(split_text) - 1 and
                                not _is_end_of_word(sub_text) and
                                not _is_start_of_word(split_text[i + 1])):
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result.append(full_word)
                            full_word = ""
                            continue

                    if tok_extended.rstrip and i > 0:
                        sub_text = sub_text.lstrip()

                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()
                else:
                    if i < len(split_text) - 1:
                        sub_text = sub_text.rstrip()

                    if i > 0:
                        sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(itertools.chain.from_iterable((self._tokenize(token)
                                                       if token not in self.unique_no_split_tokens else [token]
                                                       for token in tokenized_text)))

        no_split_token = self.unique_no_split_tokens
        return split_on_tokens(no_split_token, text)

    def truncate_sequences(self,
                           ids: List[int],
                           ids_pair: Optional[List[int]] = None,
                           num_tokens_to_remove: int = 0,
                           truncation_strategy: TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
                           stride: int = 0,
                           ) -> Tuple[List[int], List[int], List[int]]:
        """
        Truncates a sequence pair in-place following the strategy.
        """
        if num_tokens_to_remove <= 0:
            return ids, ids_pair, []

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):
                if ids_pair is None or len(ids) > len(ids_pair):
                    if not overflowing_tokens:
                        window_len = min(len(ids), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(ids[-window_len:])
                    ids = ids[:-1]
                else:
                    if not overflowing_tokens:
                        window_len = min(len(ids_pair), stride + 1)
                    else:
                        window_len = 1
                    overflowing_tokens.extend(ids_pair[-window_len:])
                    ids_pair = ids_pair[:-1]
        elif truncation_strategy == TruncationStrategy.ONLY_FIRST:
            if len(ids) > num_tokens_to_remove:
                window_len = min(len(ids), stride + num_tokens_to_remove)
                overflowing_tokens = ids[-window_len:]
                ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == TruncationStrategy.ONLY_SECOND and ids_pair is not None:
            if len(ids_pair) > num_tokens_to_remove:
                window_len = min(len(ids_pair), stride + num_tokens_to_remove)
                overflowing_tokens = ids_pair[-window_len:]
                ids_pair = ids_pair[:-num_tokens_to_remove]
        return ids, ids_pair, overflowing_tokens

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        The number of added special tokens.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0: List[int],
                                             token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create `token type ids` corresponding to the sequences passed.
        """
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def build_inputs_with_special_tokens(self,
                                         token_ids_0: List[int],
                                         token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence.
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def _pad(self,
             encoded_inputs: Dict[str, List[int]],
             max_length: Optional[int] = None,
             padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD) -> Dict[str, List[int]]:
        """
        Pad a sequence of ids to `max_length`
        """
        return_attention_mask = "attention_mask" in self.model_input_names
        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        if needs_to_be_padded:
            padded_length = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [0] * padded_length
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (encoded_inputs["token_type_ids"] +
                                                        [self.pad_token_type_id] * padded_length)
                encoded_inputs[self.model_input_names[0]] = required_input + [self.convert_tokens_to_ids
                                                                              (str(self.pad_token))] * padded_length
            elif self.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * padded_length + [1] * len(required_input)
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [self.convert_tokens_to_ids(str(self.pad_token))] \
                                                       * padded_length + encoded_inputs["token_type_ids"]
                encoded_inputs[self.model_input_names[0]] = [self.convert_tokens_to_ids(
                    str(self.pad_token))] * padded_length + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)
        return encoded_inputs

    def _tokenize(self, text: str) -> List[str]:
        """
        Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.
        """
        raise NotImplementedError

    def _convert_token_to_id(self, token: str) -> int:
        """
        Converts a token in a integer id.
        """
        raise NotImplementedError

    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts a id in a token.
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, vocab_file: str, **kwargs):
        """
        Instantiate a `PreTrainedTokenizer` from `vocab_file`.
        """
        raise NotImplementedError
