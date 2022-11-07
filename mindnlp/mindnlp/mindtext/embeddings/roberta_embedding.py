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
"""Roberta Embedding."""
import logging
from typing import Tuple
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindtext.modules.encoder.roberta import RobertaModel, RobertaConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobertaEmbedding(nn.Cell):
    """
    This is a class that loads pre-trained weight files into the model.
    """
    def __init__(self, roberta_config: RobertaConfig, is_training: bool = False):
        super(RobertaEmbedding, self).__init__()
        self.roberta = RobertaModel(roberta_config, is_training)

    def initRobertaModel(self,roberta):
        """
        Manual initialization RobertaModel
        """
        self.roberta=roberta

    def from_pretrain(self, ckpt_file):
        """
        Load the model parameters from checkpoint
        """
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self.roberta, param_dict)

    def construct(self, input_ids: Tensor, input_mask: Tensor)-> Tuple[Tensor, Tensor]:
        """
        Returns the result of the model after loading the pre-training weights

        Args:
            input_ids:A vector containing the transformation of characters into corresponding ids.
            input_mask:the mask for input_ids.

        Returns:
            sequence_output:the sequence output .
            pooled_output:the pooled output of first token:cls..
        """
        sequence_output, pooled_output, _ = self.roberta(input_ids, input_mask)
        return sequence_output, pooled_output
