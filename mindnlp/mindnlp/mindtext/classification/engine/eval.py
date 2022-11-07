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
MindText Classification eval script.
"""
from sklearn.metrics import classification_report

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.metrics import Accuracy

from ...embeddings.static_embedding import StaticEmbedding
from ...dataset.classification.yelp import YelpFullDataset
from ...classification.models.dpcnn import DPCNN
from ...common.utils.config import parse_args, Config
from ...classification.models import Model
from ...common.optimizer.builder import build_optimizer
from ...common.loss.builder import build_loss


def main(pargs):
    """
    eval function
    """
    # set config context
    config = Config(pargs.config)
    context.set_context(**config.context)
    yelp_full = YelpFullDataset(tokenizer='spacy', lang='en', batch_size=config.train.batch_size)
    dataloader = yelp_full()
    embedding = StaticEmbedding(yelp_full.vocab, model_dir_or_name=config.model.embedding)

    # set network, loss, optimizer
    network = DPCNN(embedding, config)
    network_loss = build_loss(config.loss)
    network_opt = build_optimizer(config.optimizer)

    # load pretrain model
    param_dict = load_checkpoint(config.valid.model_ckpt)
    load_param_into_net(network, param_dict)

    # init the whole Model
    model = Model(network, network_loss, network_opt, metrics={"Accuracy": Accuracy()})

    # begin to eval
    print(f'[Start eval `{config.model_name}`]')
    print("=" * 80)
    acc, target_sens, predictions = model.eval(dataloader['test'])
    result_report = classification_report(target_sens, predictions,
                                          target_names=[str(i) for i in range(int(config.model.decoder.num_class))])
    print("********Accuracy: ", acc)
    print(result_report)
    print(f'[End of eval `{config.model_name}`]')


if __name__ == '__main__':
    args = parse_args()
    main(args)
