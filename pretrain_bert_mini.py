import os
import logging
from tqdm import tqdm
from icecream import ic

import mindspore
import mindspore.nn as nn

import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import ms_function, log, mutable
from mindspore.ops import value_and_grad
from mindspore.amp import all_finite

from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from src.bert_output import BertForPretraining
from src.config import BertConfig
from src.optimizer import BertAdam
from src.metric import metric_fn
from src.dataset import read_bert_pretrain_mindrecord
from src.utils import save_bert_min_checkpoint

# Log set
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    filename='save_and_eval_10_11.log',
                    filemode='w',
                    format='%(asctime)s %(filename)s %(levelname)s \n%(message)s',
                    datefmt='%a %d %b %Y %H:%M:%S')

# # load ckpt func
# def load_bert_min_checkpoint(ckpt_file):
#     bert_min_model_dict = mindspore.load_checkpoint(ckpt_file)
#     config = BertConfig()
#     bert_min_model_to_eval = BertForPretraining(config)
#     bert_min_model_to_eval = mindspore.load_param_into_net(bert_min_model_to_eval, bert_min_model_dict)
#     return bert_min_model_to_eval

def train_loop(model=None, current_epoch=None, mode='graph', train_dataset=None, total=None, batch_size=None, optim=None, loss_fn=None):
    # temp flag
    flag_loss_nan_test = True
    # Define forward and grad function.
    # def forward_fn(*args):
    #     (total_loss, _, _), masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
    #         bert_min_model(*args)
    #     return total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs
    def forward_fn(input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_label):
        outputs, masked_lm_log_probs, next_sentence_log_probs, vocab_size = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        masked_lm_loss = loss_fn(masked_lm_log_probs.view(-1, vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = loss_fn(next_sentence_log_probs.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss, masked_lm_loss, next_sentence_loss
    grad_fn = value_and_grad(forward_fn, None, optim.parameters)

    def train_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label):
        """
        Step function for `pynative` mode.
        """
        (total_loss, masked_lm_loss, next_sentence_loss), grads = \
            grad_fn(input_ids, input_mask, segment_ids, input_ids, next_sentence_label)
        grads = grad_reducer(grads)
        optim(grads)
        return total_loss, masked_lm_loss, next_sentence_loss, grads
        
    @ms_function
    def train_step_graph(input_ids, input_mask, segment_ids, next_sentence_label):
        """
        Step function for `graph` mode.
        """
        (total_loss, masked_lm_loss, next_sentence_loss), grads = \
            grad_fn(input_ids, input_mask, segment_ids, input_ids, next_sentence_label)
        grads = grad_reducer(grads)
        if all_finite(grads):
            optim(grads)
        return total_loss, masked_lm_loss, next_sentence_loss, grads

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % (current_epoch+1))
        loss_total = 0
        mlm_loss_total = 0
        nsp_loss_total = 0
        cur_step_nums = 0
        for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
            next_sentence_label, segment_ids in train_dataset.create_tuple_iterator():
            logging.info("current steps {} \n".format(cur_step_nums))
            # ic(input_ids, input_mask, next_sentence_label, segment_ids)
            if mode == 'pynative':
                bert_loss, masked_lm_loss, next_sentence_loss, grads = train_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label)
            elif mode == 'graph':
                bert_loss, masked_lm_loss, next_sentence_loss, grads = train_step_graph(input_ids, input_mask, segment_ids, next_sentence_label)
            else:
                raise Exception('Mode Error! Only support `graph` or `pynative` mode.')
            
            # ic(grads)
            # ic(input_ids, input_mask, next_sentence_label, segment_ids)
            # outputs, masked_lm_log_probs, next_sentence_log_probs, vocab_size = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            # masked_lm_loss = loss_fn(masked_lm_log_probs.view(-1, vocab_size), input_ids.view(-1))
            # next_sentence_loss = loss_fn(next_sentence_log_probs.view(-1, 2), next_sentence_label.view(-1))
            # ic(masked_lm_log_probs.view(-1, vocab_size), input_ids.view(-1), next_sentence_log_probs.view(-1, 2), next_sentence_label.view(-1))

            # if bert_loss == float('nan'):
            #     ic(grads)
            #     ic(input_ids, input_mask, next_sentence_label, segment_ids)
            #     ic(masked_lm_log_probs.view(-1, vocab_size), input_ids.view(-1), next_sentence_log_probs.view(-1, 2), next_sentence_label.view(-1))
            #     logging.error("ocurr nan situation in {} step!!".format(cur_step_nums))
            #     break

            cur_step_nums += 1
            loss_total = loss_total + bert_loss
            mlm_loss_total = mlm_loss_total + masked_lm_loss
            nsp_loss_total = nsp_loss_total + next_sentence_loss
            # saving ckpt per 1000 steps or last step
            if config.do_save_ckpt:
                # ------loss emerge nan situation in 1000000 loop ckpt test-----
                if cur_step_nums * config.train_batch_size >= 1000000 and flag_loss_nan_test == True:
                        flag_loss_nan_test = False
                        logging.info("saving ckpt in 1000000 loop ckpt test")
                        save_bert_min_checkpoint(cur_epoch_nums=current_epoch,
                                            cur_step_nums=cur_step_nums,
                                            save_checkpoint_path=config.save_ckpt_path,
                                            rank_num=rank_id,
                                            network=bert_min_model
                                            )
                        logging.info("done")
                if cur_step_nums % 1000 == 0 or cur_step_nums == total - 1 :
                    if cur_step_nums == total - 1:
                        logging.info("********--saving ckpt with last step{}_epoch,in card{}--********\n".format(cur_step_nums, current_epoch, rank_id))
                    else:
                        logging.info("----------saving ckpt in step{}_epoch{},in card{}----------\n".format(cur_step_nums, current_epoch, rank_id))
                    save_bert_min_checkpoint(cur_epoch_nums=current_epoch,
                                            cur_step_nums=cur_step_nums,
                                            save_checkpoint_path=config.save_ckpt_path,
                                            rank_num=rank_id,
                                            network=bert_min_model
                                            )
            loss=loss_total / cur_step_nums
            #mlm_loss = mlm_loss_total / cur_step_nums
            #nsp_loss = nsp_loss_total / cur_step_nums
            t.set_postfix({'loss' : loss})
            t.update(batch_size)
            # step end
    # train epoch end
    t.close()
    return loss


def eval_loop(model=None, mode='graph', eval_dataset=None, total=None, batch_size=None, loss_fn=None):
    def eval_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label):
        """
        Step function for `pynative` mode.
        """
        outputs, masked_lm_log_probs, next_sentence_log_probs, vocab_size = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        masked_lm_loss = loss_fn(masked_lm_log_probs.view(-1, vocab_size), input_ids.view(-1))
        next_sentence_loss = loss_fn(next_sentence_log_probs.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs
        
    @ms_function
    def eval_step_graph(input_ids, input_mask, segment_ids, next_sentence_label):
        """
        Step function for `graph` mode.
        """
        outputs, masked_lm_log_probs, next_sentence_log_probs, vocab_size = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        masked_lm_loss = loss_fn(masked_lm_log_probs.view(-1, vocab_size), input_ids.view(-1))
        next_sentence_loss = loss_fn(next_sentence_log_probs.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs

    with tqdm(total=total) as t:
        t.set_description('Eval Epoch')
        loss_total = 0
        cur_step_nums = 0
        total_mlm_acc = 0
        total_nsp_acc = 0
        for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
            next_sentence_label, segment_ids in eval_dataset.create_tuple_iterator():
            logging.info("current steps {} \n".format(cur_step_nums))
            if mode == 'pynative':
                bert_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
                    eval_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label)
            elif mode == 'graph':
                bert_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
                    eval_step_graph(input_ids, input_mask, segment_ids, next_sentence_label)
            else:
                raise Exception('Mode Error! Only support `graph` or `pynative` mode.')
            eval_results = metric_fn(masked_lm_loss, masked_lm_log_probs, masked_lm_ids,
                                    masked_lm_weights, masked_lm_positions, next_sentence_loss, 
                                    next_sentence_log_probs, next_sentence_label)
            eval_results = tuple(eval_results.values())
            total_mlm_acc = total_mlm_acc + eval_results[0].asnumpy()
            total_nsp_acc = total_nsp_acc + eval_results[2].asnumpy()
            cur_step_nums += 1
            loss_total = loss_total + bert_loss
            loss=loss_total/cur_step_nums
            mlm_acc = total_mlm_acc/cur_step_nums
            nsp_acc = total_nsp_acc/cur_step_nums
            t.set_postfix(loss=loss)
            t.update(batch_size)
            # step end
    # train epoch end
    t.close()
    return loss, eval_results, mlm_acc, nsp_acc

    

def run(model=None, mode="graph", do_train=True, do_eval=False, train_dataset=None, eval_dataset=None,
        epochs=1, train_batch_size=1, eval_batch_size=1, optim=None, loss_fn=None):
    # Train
    if do_train:
        logging.info("Train Begin")
        logging.info(f"Train batch size is {train_batch_size}")
        train_total = train_dataset.get_dataset_size()
        train_dataset = train_dataset.batch(train_batch_size)
        for epoch in range(0, epochs):
            logging.info(f"Epoch {epoch+1}\n-------------------------------")
            train_loop(model, epoch, mode, train_dataset, train_total, train_batch_size, optim, loss_fn)
        logging.info('Train Done!')
                

    # Eval(After train)
    if do_eval:
        # ckpt_path = '/data0/bert/model_save/card_id_7/bert-min_ckpt_epoch_0_step_3000_card_id_7.ckpt'
        # eval_parameters = mindspore.load_checkpoint(ckpt_path)
        # bert_min_model = mindspore.load_param_into_net(bert_min_model, eval_parameters)
        logging.info("Evaluation Begin")
        logging.info(f"Eval batch size is {eval_batch_size}")
        eval_total = eval_dataset.get_dataset_size()
        eval_dataset = eval_dataset.batch(eval_batch_size)
        loss, eval_results, mlm_acc, nsp_acc = eval_loop(model, mode, eval_dataset, eval_total, eval_batch_size, loss_fn)
        mlm_loss = eval_results[1].asnumpy()
        nsp_loss = eval_results[3].asnumpy()
        logging.info('*****Eval Results*****')
        logging.info(f"global steps =  {total // eval_batch_size}")
        logging.info(f"loss = {loss}")
        logging.info(f"masked_lm_accuracy = {mlm_acc}")
        logging.info(f"masked_lm_loss = {mlm_loss}")
        logging.info(f"next_sentence_accuracy = {nsp_acc}")
        logging.info(f"next_sentence_loss = {nsp_loss}")
        # ic(mlm_acc, mlm_loss, nsp_acc, nsp_loss)



if __name__ == '__main__':
    # mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE)
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="GPU")
    init("nccl")
    # parallel test
    rank_id = get_rank()
    rank_size = get_group_size()
    ic(rank_id,rank_size)
    # parallel set
    mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL,
                                        device_num = rank_size,
                                        gradients_mean = True)
    # Define batch size and epochs.
    config = BertConfig()
    train_batch_size = config.train_batch_size
    eval_batch_size = config.eval_batch_size
    epochs = config.epochs

    # Define model.
    bert_min_model = BertForPretraining(config)
    
    # Read pre-train dataset.
    dataset_path = config.dataset_mindreocrd_dir
    # dataset_path = '/data0/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_aa/wiki_00.mindrecord'
    train_dataset = ds.MindDataset(dataset_files=dataset_path, num_shards=rank_size, shard_id=rank_id)
    eval_dataset = train_dataset
    total = train_dataset.get_dataset_size()

    # Define loss & optimizer(trick: warm up).
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    optim = BertAdam(bert_min_model.trainable_params(), lr=5e-5, warmup=0.16, t_total=total//train_batch_size)
    #optim = BertAdam(bert_min_model.trainable_params(), lr=0.1, t_total=total//train_batch_size)
    
    mean = _get_gradients_mean()
    degree = _get_device_num()
    grad_reducer = nn.DistributedGradReducer(optim.parameters, mean, degree)

    # Pretrain
    # run(model=bert_min_model, mode="pynative", do_train=True, do_eval=True, train_dataset=train_dataset, eval_dataset=eval_dataset,
    #     epochs=epochs, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, optim=optim, loss_fn=loss_fn)
    # temp flag
    flag_loss_nan_test = True
    run(model=bert_min_model, mode="graph", do_train=True, do_eval=True, train_dataset=train_dataset, eval_dataset=eval_dataset,
        epochs=epochs, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, optim=optim, loss_fn=loss_fn)
