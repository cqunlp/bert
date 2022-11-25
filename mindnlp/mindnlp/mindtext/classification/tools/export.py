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
MindText Classification export script.
"""
import os
import numpy as np

from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from mindtext.classification.utils import get_config, parse_args
from mindtext.classification.models import build_model, FastTextInferCell


def main(pargs):
    """
    eval function
    """
    # set config context
    config = get_config(pargs.config_path, overrides=pargs.override)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)

    # set network
    network = build_model(config)

    # load pretrain model
    param_dict = load_checkpoint(config.VALID.model_ckpt)
    load_param_into_net(network, param_dict)

    # init the whole Model
    ft_infer = FastTextInferCell(network)
    batch_size = config.VALID.batch_size
    if config.device_target == 'GPU':
        batch_size = config.TRAIN.distribute_batch_size_gpu
    src_tokens_shape = [batch_size, config.PREPROCESS.max_len]
    src_tokens_length_shape = [batch_size, 1]

    file_name = os.path.join(os.getcwd(), config.PREPROCESS.mid_dir_path, config.EXPORT.file_name)
    src_tokens = Tensor(np.ones((src_tokens_shape)).astype(np.int32))
    src_tokens_length = Tensor(np.ones((src_tokens_length_shape)).astype(np.int32))
    export(ft_infer, src_tokens, src_tokens_length, file_name=file_name, file_format=config.EXPORT.file_format)


if __name__ == '__main__':
    args = parse_args()
    main(args)
