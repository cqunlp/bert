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
MindText Classification infer script.
"""
import os
import spacy
import pandas as pd
import numpy as np

from mindspore import context, Model, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindtext.classification.utils import get_config, parse_args
from mindtext.classification.models import build_model, FastTextInferCell
from mindtext.classification.dataset import FastTextDataPreProcess


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
    model = Model(FastTextInferCell(network))

    spacy_nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
    spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

    process = FastTextDataPreProcess(ngram=2)
    process.read_vocab_txt(os.path.join(os.getcwd(), config.PREPROCESS.vocab_file_path))
    data = pd.read_csv(config.INFER.data_path, names=["idx", "seq1", "seq2"])
    data.fillna("", inplace=True)
    # begin to infer
    print(f'[Start infer `{config.model_name}`]')
    print("=" * 80)
    for i in zip(data["seq1"], data["seq2"], data["idx"]):
        input_sequence = process.input_preprocess(i[0], i[1], spacy_nlp, False)
        output = model.predict(Tensor(np.expand_dims(np.array(input_sequence, dtype=np.int32), 0)),
                               Tensor(np.expand_dims(np.array(len(input_sequence), dtype=np.int32), 0)))
        print(output)
    print(f'[End of infer `{config.model_name}`]')


if __name__ == '__main__':
    args = parse_args()
    main(args)
