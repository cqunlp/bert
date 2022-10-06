import logging
from tqdm import tqdm

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import ms_function, log, mutable
from mindspore.ops import value_and_grad

from src.bert_multiloss import BertForPretraining
from src.config import BertConfig
from src.optimizer import BertAdam
from src.metric import metric_fn

logging.getLogger().setLevel(logging.INFO)

from icecream import ic

    

def run(model=None, mode="graph", do_train=True, do_eval=False, train_dataset=None, eval_dataset=None,
        epochs=1, train_batch_size=1, eval_batch_size=1, optim=None):
    
    # Define forward and grad function.
    def forward_fn(*args):
        (total_loss, _, _), masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
            model(*args)
        return total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs
    grad_fn = value_and_grad(forward_fn, None, optim.parameters)

    def _run_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,
                       masked_lm_weights, masked_lm_positions, do_train=True):
        """
        Step function for `pynative` mode.
        """
        if do_train:
            (total_loss, _, _, _, _), grads = \
                grad_fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
            optim(grads)
            eval_results = {}
        else:
            total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
                forward_fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
            eval_results = metric_fn(masked_lm_loss, masked_lm_log_probs, masked_lm_ids,
                                    masked_lm_weights, masked_lm_positions, next_sentence_loss,
                                    next_sentence_log_probs, next_sentence_label)
            eval_results = tuple(eval_results.values())
        return total_loss, eval_results
        
    @ms_function
    def _run_step_graph(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,
                        masked_lm_weights, masked_lm_positions, do_train=True):
        """
        Step function for `graph` mode.
        """
        if do_train:
            (total_loss, _, _, _, _), grads = \
                grad_fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
            optim(grads)
            eval_results = {}
        else:
            total_loss, masked_lm_loss, next_sentence_loss, masked_lm_log_probs, next_sentence_log_probs = \
                forward_fn(input_ids, input_mask, segment_ids, None, None, input_ids, next_sentence_label)
            eval_results = metric_fn(masked_lm_loss, masked_lm_log_probs, masked_lm_ids,
                                    masked_lm_weights, masked_lm_positions, next_sentence_loss, 
                                    next_sentence_log_probs, next_sentence_label)
            
        return total_loss, eval_results


    def _run_epoch(mode='graph', dataset=None, total=None, batch_size=None, do_train=True):
        with tqdm(total=total) as t:
            loss_total = 0
            total_mlm_acc = 0
            total_nsp_acc = 0
            cur_step_nums = 0
            for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
                next_sentence_label, segment_ids in dataset.create_tuple_iterator():
                if mode == 'pynative':
                    loss, eval_results = \
                        _run_step_pynative(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,\
                                        masked_lm_weights, masked_lm_positions, do_train)
                elif mode == 'graph':
                    loss, eval_results = \
                        _run_step_graph(input_ids, input_mask, segment_ids, next_sentence_label, masked_lm_ids,\
                                        masked_lm_weights, masked_lm_positions, do_train)
                else:
                    raise Exception('Mode Error! Only support `graph` or `pynative` mode.')
                cur_step_nums += 1
                loss_total = loss_total + loss
                t.set_postfix(loss=loss_total/cur_step_nums)
                t.update(batch_size)
                if not do_train:
                    total_mlm_acc = total_mlm_acc + eval_results[0].asnumpy()
                    total_nsp_acc = total_nsp_acc + eval_results[2].asnumpy()
                    mlm_acc = total_mlm_acc/cur_step_nums
                    nsp_acc = total_nsp_acc/cur_step_nums
                # step end
        # train epoch end
        t.close()
        if do_train:
            return loss
        else:
            return loss, eval_results, mlm_acc, nsp_acc
    
    # Train
    if do_train:
        logging.info("Train Begin")
        logging.info(f"Train batch size is {train_batch_size}")
        train_total = train_dataset.get_dataset_size()
        train_dataset = train_dataset.batch(train_batch_size)
        for epoch in range(0, epochs):
            logging.info(f"Epoch {epoch+1}\n-------------------------------")
            _run_epoch(mode, train_dataset, train_total, train_batch_size, True)
        logging.info('Train Done!')
                

    # Eval(After train)
    if do_eval:
        logging.info("Evaluation Begin")
        logging.info(f"Eval batch size is {eval_batch_size}")
        eval_dataset = eval_dataset.batch(eval_batch_size)
        eval_total = eval_dataset.get_dataset_size()
        loss, eval_results, mlm_acc, nsp_acc = _run_epoch(mode, eval_dataset, eval_total, eval_batch_size, False)
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
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE)
    # 0. Define batch size and epochs.
    epochs = 1
    train_batch_size = 1
    eval_batch_size = 1
    # 1. Define model.
    config = BertConfig()
    model = BertForPretraining(config)
    # 2. Read pre-train dataset.
    dataset_path = '/data0/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_aa/wiki_00.mindrecord'
    train_dataset = ds.MindDataset(dataset_files=dataset_path)
    total = train_dataset.get_dataset_size()
    eval_dataset = ds.MindDataset(dataset_files=dataset_path)
    # 3. Define optimizer(trick: warm up).
    optim = BertAdam(model.trainable_params(), lr=0.1, t_total=total/train_batch_size)
    # 6. Pretrain
    # run(model=model, mode="pynative", do_train=True, do_eval=True, train_dataset=train_dataset, eval_dataset=eval_dataset,
    #     epochs=1, train_batch_size=32, eval_batch_size=32, optim=optim)
    run(model=model, mode="graph", do_train=True, do_eval=True, train_dataset=train_dataset, eval_dataset=eval_dataset,
        epochs=1, train_batch_size=32, eval_batch_size=32, optim=optim)
