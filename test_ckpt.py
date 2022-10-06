import os
import mindspore
import mindspore.dataset as ds
from mindspore import Tensor,ops
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
    mindspore.save_checkpoint(network, ckpt_save_dir)

def train(mode='graph'):
    # 6. train
    for epoch in range(0, epochs):
        # epoch begin
        with tqdm(total=total) as t:
            t.set_description('Epoch %i' % epoch)
            loss_total = 0
            cur_step_nums = 0
            # step begin
            for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
            next_sentence_label, segment_ids in train_dataset.create_tuple_iterator():
                if mode == 'pynative':
                    loss = _train_step(input_ids, input_mask, segment_ids, next_sentence_label)
                elif mode == 'graph':
                    loss = _train_step_graph(input_ids, input_mask, segment_ids, next_sentence_label)
                else:
                    log.warning('Mode Error!')

                # ic(type(loss))
                loss_total = loss_total + loss

                # save and choose checkpoint 
                # saving ckpt per 1000 steps
                if cur_step_nums % 1000 == 0:
                    ("----------saving ckpt in epoch{}_step{}----------".format(epoch, cur_step_nums))
                    save_bert_min_checkpoint(cur_epoch_nums=epoch,
                                             cur_step_nums=cur_step_nums,
                                             save_checkpoint_path='/data0/bert/model_save',
                                             rank_num=rank_id,
                                             network=bert_min_model
                                             )
                    print("-----------------saved ckpt---------------------")
                cur_step_nums += 1
                t.set_postfix(loss=loss_total/cur_step_nums)
                t.update(batch_size)
                # step end
        # train epoch end
        t.close()


if __name__ == '__main__':
    ckpt1='/data0/bert/model_save/card_id_0/bert-min_ckpt_epoch_0_step_0_card_id_0.ckpt'
    ckpt2='/data0/bert/model_save/card_id_0/bert-min_ckpt_epoch_0_step_700_card_id_0.ckpt'
    bert1_dict = mindspore.load_checkpoint(ckpt1)
    bert2_dict = mindspore.load_checkpoint(ckpt2)
    # bert = BertForPretraining(config)
    print(bert1_dict)
    b1 = bert1_dict['bert.encoder.layer.0.attention.self_attn.query.weight']
    b2 = bert2_dict['bert.encoder.layer.0.attention.self_attn.query.weight']
    print(Tensor(b1))  
    print(mindspore.ops.equal(Tensor(b1),Tensor(b2)))
    print(Tensor(b2))



    