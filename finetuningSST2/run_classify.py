import sys
import os
import argparse
from tqdm import tqdm
from typing import Optional
sys.path.append("..")
import mindspore
from mindspore import context
from mindspore import nn
from mindspore import ops
from mindspore import dataset
from src.config import BertConfig
from src.optimization import BertLearningRate
from model import BertBinaryClassificationModel
from mindnlp.mindnlp.mindtext.dataset.classification.sst import SST2Dataset

def getpwd():
    pwd = sys.path[0]
    if os.path.isfile(pwd):
        pwd = os.path.dirname(pwd)
    return pwd

def get_sst2_dataset(dataset_path, bacth_size, max_length):
    dataset = SST2Dataset(paths=dataset_path,
                      tokenizer="bert-base-uncased",
                      max_length=max_length,
                      truncation_strategy=True,
                      columns_list=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
                      test_columns_list=['input_ids', 'token_type_ids', 'attention_mask'],
                      batch_size=bacth_size)
    sst_2_ds = dataset()
    train_dataset = sst_2_ds['train']
    # test_dataset haven't label cause it only work for interface
    test_dataset = sst_2_ds['dev']
    # train_dataset_path = "{data_path}/SST2_mr/sst_2_train_data.mindrecord".format(data_path=dataset_path)
    # test_dataset_path = "{data_path}/SST2_mr/sst_2_test_data.mindrecord".format(data_path=dataset_path)
    # train_dataset = dataset.MindDataset(dataset_files=train_dataset_path)
    # test_dataset = dataset.MindDataset(dataset_files=test_dataset_path)
    # train_dataset = train_dataset.batch(bacth_size, drop_remainder=True)
    # test_dataset = test_dataset.batch(bacth_size, drop_remainder=True)
    return train_dataset, test_dataset

def init_sst2_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt", type=str, required=True,\
                        help="Choose pretrain bert ckpt path.")
    parser.add_argument("--dataset_path", default='SST-2', type=str,\
                        help="Choose dataset path.")
    parser.add_argument("--config", type=str, required=True,\
                        help="Choose bert config file.")
    # parser.add_argument("--output", default= os.path.join(getpwd(), "outputs"), type=str,\
    #                     help="Choose outputs path.")
    parser.add_argument("--batch_size", default=16, type=int, required=True,\
                        help="Choose batch size.")
    parser.add_argument("--epochs", default=10, type=int, required=True,\
                        help="Choose training epochs value.")
    parser.add_argument("--lr", default=2e-5, type=float, required=True,\
                        help="Choose learning rate.")
    parser.add_argument("--max_length", default=64, type=int,\
                        help="Choose max length.")
    parser.add_argument("--acc", default=85, type=float, \
                        help="Choose accuracy need to save.")
    args = parser.parse_args()
    return args

def test_loop(model,
              dataset,
              output_file: Optional[str] = None,
              current_step: Optional[int] = None,
              current_epoch: Optional[int] = None,
              test_flag=False):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for input_ids, attention_mask, token_type_ids, label in dataset.create_tuple_iterator():
        loss, logits = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        label=label)
        total += len(input_ids)
        test_loss += loss.asnumpy()
        correct_loop = (logits.argmax(1) == label.view(-1)).asnumpy().sum()
        correct += correct_loop
    test_loss /= num_batches
    correct /= total
    # choose better accuracy ckpt
    if test_flag is False:
        # 如果验收精度达标就保存这个fine-tuning的checkpoint accuracy可以自己定
        if correct*100 >= args.acc:
            output_ckpt_file = os.path.join(output_file, "{epoch}_epoch_{step}_step_acc_{acc}%.ckpt".format(
                epoch=current_epoch, step=current_step, acc=(correct*100)
            ))
            mindspore.save_checkpoint(model, output_ckpt_file)
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

def train_and_test(model, train_dataset, dev_dataset, optimizer, current_epoch):
    output_file = os.path.join(getpwd(), "outputs")
    if not os.path.exists(output_file):
        try:
            os.mkdir(output_file)
        except FileExistsError:
            pass
    def forward_fn(input_ids, attention_mask, token_type_ids, label):
        loss, logits = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        label=label)
        return loss, logits
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(input_ids, attention_mask, token_type_ids, label):
        (loss, _), grads = grad_fn(input_ids, attention_mask, token_type_ids, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = train_dataset.get_dataset_size()
    batch_size = train_dataset.get_batch_size()
    model.set_train(True)

    with tqdm(total=size * batch_size) as t:
        for batch, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_dataset.create_tuple_iterator()):
            t.set_description('Epoch %i' % current_epoch)
            loss = train_step(input_ids, attention_mask, token_type_ids, label)
            loss, current = loss.asnumpy(), batch
            t.set_postfix(loss=loss)
            # if batch % 10 == 0:
            #     loss, current = loss.asnumpy(), batch
            #     print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
            # add test
            if batch % 100 == 0:
                print("In epoch {} _ batch {} testing".format(current_epoch, current))
                test_loop(model, dev_dataset, output_file, batch, current_epoch, False)
            t.update(batch_size)

def check_save_sst2_ckpt(test_dataset, config, sst2_ckpt):
    test_model = BertBinaryClassificationModel(config)
    test_bert_dict = mindspore.load_checkpoint(sst2_ckpt)
    mindspore.load_param_into_net(test_model, test_bert_dict)
    acc = test_loop(model=test_model,
                    dataset=test_dataset,
                    test_flag=True)
    print('The ckpt acc is: ',acc)

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    # context.set_context(mode=context.PYNATIVE_MODE)
    # train set
    args = init_sst2_args()
    epoch_num = args.epochs
    ckpt_file = args.bert_ckpt
    config = BertConfig(args.config)
    bert_sst_2 = BertBinaryClassificationModel(config, ckpt_file)
    params = bert_sst_2.trainable_params()
    # get datset
    try:
        train_dataset, test_dataset = get_sst2_dataset(args.dataset_path, args.batch_size, args.max_length)
    except ModuleNotFoundError:
        os.system('pip install -r ../requirements.txt')
        train_dataset, test_dataset = get_sst2_dataset(args.dataset_path, args.batch_size, args.max_length)

    lr_schedule = BertLearningRate(learning_rate=args.lr,
                                   end_learning_rate=0.0,
                                   warmup_steps=int(train_dataset.get_dataset_size() * epoch_num * 0.1),
                                   decay_steps=train_dataset.get_dataset_size() * epoch_num,
                                   power=1.0)
    optimizer = nn.AdamWeightDecay(params, lr_schedule, eps=1e-8)
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_and_test(bert_sst_2, train_dataset, test_dataset, optimizer, epoch+1)
    print("Done!")
    # only test for fine-tuning
    # check_save_sst2_ckpt(test_dataset, config,'outputs/4_test_4_epoch_1000_step_acc_82.91284403669725%.ckpt')
