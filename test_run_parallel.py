import os
import logging

import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds

from icecream import ic
from tqdm import tqdm

from mindspore import nn
from mindspore import ms_function, log, mutable
from mindspore.ops import value_and_grad
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from src.bert_multiloss import BertForPretraining
from src.config import BertConfig
from src.optimizer import BertAdam
from src.metric import metric_fn
from src.dataset import read_bert_pretrain_mindrecord

def _run_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,
                       masked_lm_weights, masked_lm_positions, fn=None, do_train=True):
    """
    Step function for `pynative` mode.
    """
    if do_train:
        (total_loss, _, _, _, _), grads = \
            fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
        grads = grad_reducer(grads)
        optim(grads)
        eval_results = {}
    else:
        total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
            fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
        eval_results = metric_fn(masked_lm_loss, masked_lm_log_probs, masked_lm_ids,
                                 masked_lm_weights, masked_lm_positions, next_sentence_loss,
                                 next_sentence_log_probs, next_sentence_label)
    return total_loss, eval_results
    
@ms_function
def _run_step_graph(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,
                    masked_lm_weights, masked_lm_positions, fn=None, do_train=True):
    """
    Step function for `graph` mode.
    """
    if do_train:
        (total_loss, _, _, _, _), grads = \
            fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
        grads = grad_reducer(grads)
        optim(grads)
        eval_results = {}
    else:
        total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
            fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
        eval_results = metric_fn(masked_lm_loss, masked_lm_log_probs, masked_lm_ids,
                                 masked_lm_weights, masked_lm_positions, next_sentence_loss, 
                                 next_sentence_log_probs, next_sentence_label)
    return total_loss, eval_results

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

def _run_epoch(this_epoch, mode='graph', dataset=None, total=None, batch_size=None, fn=None, do_train=True):
    with tqdm(total=total) as t:
        loss_total = 0
        total_mlm_acc = 0
        total_nsp_acc = 0
        cur_step_nums = 0
        for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
            next_sentence_label, segment_ids in dataset.create_tuple_iterator():
            logging.log(msg="curr steps {} and total step in epoch{} is {}\n".format(cur_step_nums, this_epoch, total),level=logging.INFO)
            if mode == 'pynative':
                loss, eval_results = \
                    _run_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,
                                       masked_lm_weights, masked_lm_positions, fn, do_train)
            elif mode == 'graph':
                loss, eval_results = \
                    _run_step_graph(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,
                                    masked_lm_weights, masked_lm_positions, fn, do_train)
            else:
                log.warning('Mode Error!')
            cur_step_nums += 1
            loss_total = loss_total + loss
            total_mlm_acc = total_mlm_acc + eval_results["masked_lm_accuracy"]
            total_nsp_acc = total_nsp_acc + eval_results["next_sentence_accuracy"]
            mlm_acc = total_mlm_acc/cur_step_nums
            nsp_acc = total_nsp_acc/cur_step_nums
            # save and choose checkpoint 
            # saving ckpt per 1000 steps or last step
            last_step = total - 1 
            if config.do_save_ckpt:
                if cur_step_nums % 1000 == 0 or cur_step_nums == last_step :
                    if cur_step_nums == last_step:
                        logging.log(msg="********--saving ckpt in epoch{} with last step{},in card{}--********\n".format(this_epoch, cur_step_nums, rank_id),level=logging.INFO)
                    else:
                        logging.log(msg="----------saving ckpt in epoch{}_step{},in card{}----------\n".format(this_epoch, cur_step_nums, rank_id),level=logging.INFO)
                    save_bert_min_checkpoint(cur_epoch_nums=this_epoch,
                                            cur_step_nums=cur_step_nums,
                                            save_checkpoint_path=config.save_ckpt_path,
                                            rank_num=rank_id,
                                            network=bert_min_model
                                            )
            t.set_postfix(loss=loss_total/cur_step_nums)
            t.update(batch_size)
            # step end
    # train epoch end
    t.close()
    return loss, eval_results, mlm_acc, nsp_acc

def forward_fn(*args):
    (total_loss, _, _), masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
        bert_min_model(*args)
    return total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs
    

def run(model=None, mode="graph", do_train=True, do_eval=False, dataset=None, epochs=40, train_batch_size=128, eval_batch_size=128, optim=None):
    # Define forward and grad function.
    grad_fn = value_and_grad(forward_fn, None, optim.parameters)
    # Train
    if do_train:
        logging.info("Train Begin")
        logging.info(f"Train batch size is {train_batch_size}")
        dataset = dataset.batch(train_batch_size)
        for epoch in range(0, epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            _run_epoch(epoch, mode, dataset, total, train_batch_size, grad_fn, True)

    # Eval(After train)
    if do_eval:
        # change context envirnment
        mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)
        mode = 'pynative'
        logging.info("Evaluation Begin")
        logging.info(f"Eval batch size is {eval_batch_size}")
        dataset = dataset.batch(eval_batch_size)
        loss, eval_results, mlm_acc, nsp_acc = _run_epoch(epoch, mode, dataset, total, eval_batch_size, forward_fn, False)
        mlm_loss = eval_results["masked_lm_loss"]
        nsp_loss = eval_results["next_sentence_loss"]
        logging.info('*****Eval Results*****')
        logging.info(f"global steps =  {int(total/eval_batch_size)}")
        logging.info(f"loss = {loss}")
        logging.info(f"masked_lm_accuracy = {mlm_acc}")
        logging.info(f"masked_lm_loss = {mlm_loss}")
        logging.info(f"next_sentence_accuracy = {nsp_acc}")
        logging.info(f"next_sentence_loss = {nsp_loss}")

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
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")
    init("nccl")
    # 0. Define batch size and epochs.
    config = BertConfig()
    train_batch_size = config.train_batch_size
    eval_batch_size = config.eval_batch_size
    epochs = config.epochs
    file_name = config.dataset_mindreocrd_dir
    file_name = '/data0/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_aa/wiki_00.mindrecord'
    # Parallel set
    rank_id = get_rank()
    rank_size = get_group_size()
    ic(rank_id,rank_size)
    # 1 & 2. Parallel read and process pre-train dataset.
    pretrain_dataset, total = read_bert_pretrain_mindrecord(batch_size=train_batch_size,\
    file_name=file_name, rank_id=rank_id, rank_size=rank_size)
    
    # 3. Define model.
    bert_min_model = BertForPretraining(config)
    
    # 4. Define optimizer.
    optim = BertAdam(bert_min_model.trainable_params(), lr=0.1, t_total=total/train_batch_size)
    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optim.parameters, mean, degree)
    
    # Pretrain
    run(model=bert_min_model, mode="pynative", do_train=True, do_eval=True, dataset=pretrain_dataset, epochs=epochs,
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, optim=optim)