import os
import logging
import mindspore
import mindspore.dataset as ds

from icecream import ic
from tqdm import tqdm
from mindspore import ms_function, log, mutable
from mindspore.ops import value_and_grad
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean
from mindspore import nn
from src.bert import BertForPretraining
from src.config import BertConfig
from src.optimizer import BertAdam
from src.dataset import read_bert_pretrain_mindrecord


def _train_step(input_ids, input_mask, segment_ids, next_sentence_label):
    loss, grads = grad_fn(input_ids, input_mask, segment_ids,
                          None, None, input_ids, next_sentence_label)
    grads = grad_reducer(grads)
    optim(grads)
    return loss

@ms_function
def _train_step_graph(input_ids, input_mask, segment_ids, next_sentence_label):
    loss, grads = grad_fn(input_ids, input_mask, segment_ids,
                          None, None, input_ids, next_sentence_label)
    grads = grad_reducer(grads)
    optim(grads)
    return loss

def save_bert_min_checkpoint(cur_epoch_nums,\
                            cur_step_nums,\
                            save_checkpoint_path,\
                            rank_num,\
                            network):
    per_card_save_model_path = ('bert-min_ckpt_'+\
    'epoch_{}_'.format(cur_epoch_nums)+\
    'step_{}_'.format(cur_step_nums)+\
    'card_id_{}'.format(rank_num))
    ckpt_save_dir = os.path.join(save_checkpoint_path,('card_id_' + str(rank_num)),\
    per_card_save_model_path)
    mindspore.save_checkpoint(save_obj=network,
                              ckpt_file_name=ckpt_save_dir,
                              integrated_save=True,
                              async_save=True)
    

def train(mode='graph'):
    # 6. train
    for epoch in range(0, epochs):
        # epoch begin
        with tqdm(total=total) as t:
            t.set_description('Epoch %i' % epoch)
            loss_total = 0
            cur_step_nums = 0
            
            # logging.log(msg="last steps {}\n".format(last_steps),level=logging.INFO)
            # step begin
            for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
            next_sentence_label, segment_ids in train_dataset.create_tuple_iterator():
                logging.log(msg="curr steps {} and total step in epoch{} is {}\n".format(cur_step_nums, epoch, total),level=logging.INFO)
                if mode == 'pynative':
                    loss = _train_step(input_ids, input_mask, segment_ids, next_sentence_label)
                elif mode == 'graph':
                    loss = _train_step_graph(input_ids, input_mask, segment_ids, next_sentence_label)
                else:
                    log.warning('Mode Error!')

                # ic(type(loss))
                loss_total = loss_total + loss
                last_step = total - 1
                # save and choose checkpoint 
                # saving ckpt per 1000 steps or last step
                if config.do_save_ckpt:
                    if cur_step_nums % 1000 == 0 or cur_step_nums == last_step :
                        if cur_step_nums == last_step:
                            logging.log(msg="********--saving ckpt in epoch{} with last step{},in card{}--********\n".format(epoch, cur_step_nums, rank_id),level=logging.INFO)
                        else:
                            logging.log(msg="----------saving ckpt in epoch{}_step{},in card{}----------\n".format(epoch, cur_step_nums, rank_id),level=logging.INFO)
                        save_bert_min_checkpoint(cur_epoch_nums=epoch,
                                                cur_step_nums=cur_step_nums,
                                                save_checkpoint_path=config.save_ckpt_path,
                                                rank_num=rank_id,
                                                network=bert_min_model
                                                )
                        # logging.log(msg="-----------------saved ckpt---------------------\n",level=logging.INFO)
                # eval last ckpt here
                
                cur_step_nums += 1
                t.set_postfix(loss=loss_total/cur_step_nums)
                t.update(batch_size)
                # step end
        # train epoch end
        t.close()


if __name__ == '__main__':
    # Log set
    logging.basicConfig(level=logging.INFO,
                    filename='save_and_eval.log',
                    filemode='w',
                    format='%(asctime)s %(filename)s %(levelname)s \n%(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S'
                    )
    # Parallel initialization
    mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL)
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
    init("nccl")
    # 0. Define batch size and epochs.
    config = BertConfig()
    batch_size = config.train_batch_size
    epochs = config.epochs
    file_name = config.dataset_mindreocrd_dir
    
    # Parallel set
    rank_id = get_rank()
    rank_size = get_group_size()
    ic(rank_id,rank_size)
    # 1 & 2. Parallel read and process pre-train dataset.
    train_dataset, total = read_bert_pretrain_mindrecord(batch_size=batch_size,\
    file_name=file_name, rank_id=rank_id, rank_size=rank_size)

    # 3. Define model.
    bert_min_model = BertForPretraining(config)
    
    # 4. Define optimizer.
    optim = BertAdam(bert_min_model.trainable_params(), lr=0.1)

    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optim.parameters, mean, degree)
    # 5. Define forward and grad function.
    def forward_fn(*args):
        (loss, _, _) = bert_min_model(*args)
        return loss
    grad_fn = value_and_grad(forward_fn, None, optim.parameters)
    # 6. Train
    train('graph')
