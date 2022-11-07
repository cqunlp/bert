# Copyright 2020 Huawei Technologies Co., Ltd
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
MindText Classification train script.
"""
import time
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import TimeMonitor
from mindspore.train.callback import Callback

from mindtext.classification.utils import get_config, parse_args
from mindtext.classification.dataset import create_dataset
from mindtext.classification.models import build_model, create_loss, create_optimizer, Model


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))


set_seed(5)


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_ids=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_ids
        self.time_stamp_first = get_ms_timestamp()

    def step_end(self, run_context):
        """Monitor the loss in training."""
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - self.time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}".format(
                time_stamp_current - self.time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs.asnumpy())))
            f.write('\n')


def main(pargs):
    """
    Train function
    """
    # set config context
    config = get_config(pargs.config_path, overrides=pargs.override)
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target)

    # load mindrecord dataset
    dataset_train = create_dataset(config, select_dataset="train")
    config.OPTIMIZER.decay_steps = dataset_train.get_dataset_size()

    # set network, loss, optimizer
    network = build_model(config)
    network_loss = create_loss(config)
    network_opt = create_optimizer(network.trainable_params(), config)

    # set callbacks for the network
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=config.TRAIN.save_ckpt_steps,
        keep_checkpoint_max=config.TRAIN.keep_ckpt_max)
    ckpt_callback = ModelCheckpoint(prefix=config.model_name,
                                    directory=config.TRAIN.save_ckpt_dir,
                                    config=ckpt_config)
    loss_monitor = LossCallBack()
    time_monitor = TimeMonitor(data_size=dataset_train.get_dataset_size())
    callbacks = [time_monitor, loss_monitor, ckpt_callback]

    # init the Trainer
    model = Model(network,
                  network_loss,
                  network_opt)

    print(f'[Start training `{config.model_name}`]')
    print("=" * 80)

    model.train(config.TRAIN.epochs,
                train_dataset=dataset_train,
                callbacks=callbacks)
    print(f'[End of training `{config.model_name}`]')


if __name__ == "__main__":
    args = parse_args()
    main(args)
