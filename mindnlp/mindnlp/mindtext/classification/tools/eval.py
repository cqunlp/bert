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

from mindtext.classification.utils import get_config, parse_args
from mindtext.classification.dataset import create_dataset
from mindtext.classification.models import build_model, create_loss, create_optimizer, Model


def main(pargs):
    """
    eval function
    """
    # set config context
    config = get_config(pargs.config_path, overrides=pargs.override)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)

    # load mindrecord dataset
    dataset_eval = create_dataset(config, select_dataset="valid")

    # set network, loss, optimizer
    network = build_model(config)
    network_loss = create_loss(config)
    network_opt = create_optimizer(network.trainable_params(), config)

    # load pretrain model
    param_dict = load_checkpoint(config.VALID.model_ckpt)
    load_param_into_net(network, param_dict)

    # init the whole Model
    model = Model(network,
                  network_loss,
                  network_opt,
                  metrics={"Accuracy": Accuracy()})

    # begin to eval
    print(f'[Start eval `{config.model_name}`]')
    print("=" * 80)
    acc, target_sens, predictions = model.eval(dataset_eval)
    result_report = classification_report(target_sens, predictions,
                                          target_names=[str(i) for i in range(int(config.MODEL_PARAMETERS.num_class))])
    print("********Accuracy: ", acc)
    print(result_report)
    print(f'[End of eval `{config.model_name}`]')


if __name__ == '__main__':
    args = parse_args()
    main(args)
