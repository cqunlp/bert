from tqdm import tqdm

import mindspore
import mindspore.dataset as ds
from mindspore import ms_function, log, mutable
from mindspore.ops import value_and_grad

from src.bert import BertForPretraining
from src.config import BertConfig
from src.optimizer import BertAdam

from icecream import ic


def train(mode='graph'):
    """
    Train function for Bert pre-training.
    """
    # 4. Define optimizer(trick: warm up).
    optim = BertAdam(model.trainable_params(), lr=0.1, t_total=total/batch_size)

    def _train_step(input_ids, input_mask, segment_ids, next_sentence_label):
        """
        Train step function for `pynative` mode.
        """
        loss, grads = grad_fn(input_ids, input_mask, segment_ids,\
        None, None, input_ids, next_sentence_label)
        optim(grads)
        return loss
    
    @ms_function
    def _train_step_graph(input_ids, input_mask, segment_ids, next_sentence_label):
        """
        Train step function for `graph` mode.
        """
        loss, grads = grad_fn(input_ids, input_mask, segment_ids,\
        None, None, input_ids, next_sentence_label)
        optim(grads)
        return loss

    # 5. Define forward and grad function.
    def forward_fn(*args):
        (loss, _, _) = model(*args)
        return loss
    grad_fn = value_and_grad(forward_fn, None, optim.parameters)
    # 6. train
    for epoch in range(0, epochs):
        # epoch begin
        print(f"Epoch {epoch+1}\n-------------------------------")
        with tqdm(total=total) as t:
            t.set_description('Epoch %i' % (epoch+1))
            loss_total = 0
            global cur_step_nums
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
                cur_step_nums += 1
                t.set_postfix(loss=loss_total/cur_step_nums)
                t.update(batch_size)
                # step end
        # train epoch end
        t.close()
    print("Done!")

if __name__ == '__main__':
    mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE)
    # 0. Define batch size and epochs.
    batch_size = 4
    epochs = 1
    # 1. Read pre-train dataset.
    train_dataset_path = '/data0/dataset/bert_pretrain/bert/src/generate_mindrecord/9_15_wiki/mr_aa/wiki_00.mindrecord'
    train_dataset = ds.MindDataset(dataset_files=train_dataset_path)
    # 2. Batchify the dataset.
    total = train_dataset.get_dataset_size()
    train_dataset = train_dataset.batch(batch_size)
    # 3. Define model.
    config = BertConfig()
    model = BertForPretraining(config)
    # 6. Pretrain
    # train('graph')
    train('graph')

