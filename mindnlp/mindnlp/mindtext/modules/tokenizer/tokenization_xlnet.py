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
    Tokenization classes for XLNet.
"""
import unicodedata
from typing import Optional, Dict, Any, List

import sentencepiece as spm

from .base_tokenizer import PreTrainedTokenizer, AddedToken
from .utils import SPIECE_UNDERLINE


class XLNetTokenizer(PreTrainedTokenizer):
    """
    XLNet tokenizer.

    Args:
        vocab_file (str): Vocab file path.
        do_lower_case (bool): Whether or not to lowercase word.
    """
    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            remove_space=True,
            keep_accents=False,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
            cls_token="<cls>",
            mask_token="<mask>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> None:
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super(XLNetTokenizer, self).__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            padding_side="left",
            **kwargs,
        )
        self._pad_token_type_id = 3

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        self.padding_side = "left"

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    def preprocess_text(self, inputs):
        """
        Preprocess text before tokenize.
        """
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.convert_tokens_to_ids(self.sep_token)]
        cls = [self.convert_tokens_to_ids(self.cls_token)]
        if token_ids_1 is None:
            return token_ids_0 + sep + cls
        return token_ids_0 + sep + token_ids_1 + sep + cls

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.convert_tokens_to_ids(self.sep_token)]
        cls_segment_id = [2]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string.
        """
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    @classmethod
    def from_pretrained(cls, vocab_file: str, **kwargs):
        return cls(vocab_file)
