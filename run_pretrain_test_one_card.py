import time
import mindspore
import mindspore.dataset as ds
from mindspore import ms_function, log, mutable
from mindspore.ops import cross_entropy
from mindspore import nn

from src.api import value_and_grad
from src.amp import auto_mixed_precision
from src.bert import BertForPretraining
from src.config import BertConfig

from tqdm import tqdm

def train(model, optimizer, train_dataset, epochs, jit=True, amp=False):
    """
    Train function for Bert pre-training.
    """
    # 5. Define forward and grad function.
    def forward_fn(input_ids, input_mask, segment_ids, \
                   masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label):
        (prediction_scores, seq_relationship_score, _) = model(input_ids, input_mask, segment_ids, None, None, masked_lm_positions)
        # ic(prediction_scores.shape) # (batch_size, 128, 30522)
        # ic(masked_lm_labels.shape) # (batch_size, 20)
        masked_lm_loss = cross_entropy(prediction_scores.view(-1, prediction_scores.shape[-1]),
                                       masked_lm_ids.view(-1), masked_lm_weights.view(-1))
        next_sentence_loss = cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        
        return total_loss, masked_lm_loss, next_sentence_loss

    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
                   next_sentence_label, segment_ids):
        (total_loss, masked_lm_loss, next_sentence_loss), grads = grad_fn(input_ids, input_mask, segment_ids, \
                              masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label)
        optimizer(grads)
        return total_loss, masked_lm_loss, next_sentence_loss

    if jit:
        train_step = ms_function(train_step)

    # 6. train
    total = train_dataset.get_dataset_size()
    for epoch in range(0, epochs):
        # epoch begin
        print(f"Epoch {epoch+1}\n-------------------------------")
        with tqdm(total=total) as t:
            t.set_description('Epoch %i' % (epoch+1))
            loss_total = 0
            cur_step_nums = 0
            # step begin
            for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
                    next_sentence_label, segment_ids in train_dataset.create_tuple_iterator():
                # print(input_ids.shape)
                # s = time.time()
                total_loss, masked_lm_loss, next_sentence_loss = train_step(input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
                    next_sentence_label, segment_ids)
                # e = time.time()
                # print(e - s)
                loss_total = loss_total + total_loss.asnumpy()
                cur_step_nums += 1
                t.set_postfix(loss=loss_total/cur_step_nums)
                t.update(1)
                # step end
                # break
        # train epoch end
        t.close()
    print("Done!")

if __name__ == '__main__':
    mindspore.set_context(enable_graph_kernel=True)
    # mindspore.set_context(mode=mindspore.GRAPH_MODE)
    # profiler = Profiler()
    # 0. Define batch size and epochs.
    batch_size = 256
    epochs = 10
    # 1. Read pre-train dataset.
    train_dataset_path = './dataset/wiki_00.mindrecord'
    train_dataset = ds.MindDataset(dataset_files=train_dataset_path)
    config = BertConfig()
    # dataset_path = config.dataset_mindreocrd_dir
    # train_dataset = ds.MindDataset(dataset_files=dataset_path, num_samples=2560)
    # 2. Batchify the dataset.
    total = train_dataset.get_dataset_size()
    train_dataset = train_dataset.batch(batch_size)
    # train_dataset = train_dataset.take(2)
    # 3. Define model.
    config = BertConfig()
    model = BertForPretraining(config)

    model = auto_mixed_precision(model, "O1")
    # 4. Define optimizer(trick: warm up).
    # optimizer = BertAdam(model.trainable_params(), lr=5e-5, warmup=0.16, t_total=total//batch_size)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=0.001)
    # optimizer = AdamWeightDecayForBert(model.trainable_params(), learning_rate=0.1)
    # 6. Pretrain
    train(model, optimizer, train_dataset, epochs, jit=True)

    # profiler.analyse()