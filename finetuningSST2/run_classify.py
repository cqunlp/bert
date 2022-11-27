import sys
import time
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
# from mindnlp.mindnlp.mindtext.dataset.classification.sst import SST2Dataset


def str2bool(str):
    return True if str.lower() == 'true' else False

def getpwd():
    pwd = sys.path[0]
    if os.path.isfile(pwd):
        pwd = os.path.dirname(pwd)
    return pwd

def get_sst2_dataset(dataset_path, train_batch_size, test_batch_size, max_length):
    # dataset = SST2Dataset(paths=dataset_path,
    #                   tokenizer="bert-base-uncased",
    #                   max_length=max_length,
    #                   truncation_strategy=True,
    #                   columns_list=['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    #                   test_columns_list=['input_ids', 'token_type_ids', 'attention_mask'],
    #                   batch_size=batch_size)
    # sst_2_ds = dataset()
    # train_dataset = sst_2_ds['train']
    # # test_dataset haven't label cause it only work for interface
    # test_dataset = sst_2_ds['dev']
    train_dataset = dataset.MindDataset(dataset_files="{data_path}-{max_len}/sst_2_train_data.mindrecord".format(data_path=dataset_path, max_len=max_length))
    test_dataset = dataset.MindDataset(dataset_files="{data_path}-{max_len}/sst_2_test_data.mindrecord".format(data_path=dataset_path, max_len=max_length))
    train_dataset = train_dataset.batch(train_batch_size)
    test_dataset = test_dataset.batch(test_batch_size)
    return train_dataset, test_dataset

def init_sst2_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_target", default='Ascend', type=str, \
                        help="Backend device.") 
    parser.add_argument("--amp", default='True', type=str2bool, required=True,\
                        help="whether use amp.")
    parser.add_argument("--bert_ckpt", type=str, required=True,\
                        help="Choose pretrain bert ckpt path.")
    parser.add_argument("--dataset_path", default='sst-2', type=str,\
                        help="Choose dataset path.")
    parser.add_argument("--config", type=str, required=True,\
                        help="Choose bert config file.")
    # parser.add_argument("--output", default= os.path.join(getpwd(), "outputs"), type=str,\
    #                     help="Choose outputs path.")
    parser.add_argument("--train_batch_size", default=16, type=int, required=True,\
                        help="Choose train batch size.")
    parser.add_argument("--test_batch_size", default=16, type=int, required=True,\
                        help="Choose test batch size.")
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
    # 👴
    with tqdm(total=num_batches * dataset.get_batch_size()) as t:
        t.set_description('Evaluating')
        for data in dataset.create_dict_iterator():
            loss, logits = model(input_ids=data['input_ids'].squeeze(-2),
                                 attention_mask=data['attention_mask'].squeeze(-2),
                                 token_type_ids=data['token_type_ids'].squeeze(-2),
                                 label=data['label'].squeeze(-2))
            total += len(data['input_ids'])
            test_loss += loss.asnumpy()
            correct_loop = (logits.argmax(1) == data['label'].view(-1)).asnumpy().sum()
            correct += correct_loop
            t.update(dataset.get_batch_size())
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

    global best_accuracy
    if best_accuracy <= 100*correct:
        best_accuracy = 100*correct
    print(f"\nTest: \n Accuracy: {(100*correct):>0.1f}%, Best-Accuracy: {best_accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return 100*correct

def train_and_test(model, loss_scaler, train_dataset, dev_dataset, optimizer, current_epoch):
    global save_time
    save_description = "epoch{}-max_len{}-train_bs{}-test_bs{}-lr{}-time{}".format(
        args.epochs,
        args.max_length,
        args.train_batch_size,
        args.test_batch_size,
        args.lr,
        save_time)
    output_file = os.path.join(getpwd(), "outputs", save_description)
    if not os.path.exists(output_file):
        try:
            os.mkdir(output_file)
        except FileNotFoundError:
            os.mkdir(os.path.join(getpwd(), "outputs"))
            os.mkdir(output_file)
    def forward_fn(input_ids, attention_mask, token_type_ids, label):
        loss, logits = model(input_ids=input_ids.squeeze(-2),
                        attention_mask=attention_mask.squeeze(-2),
                        token_type_ids=token_type_ids.squeeze(-2),
                        label=label.squeeze(-2))
        # 混合精度加这一条
        loss = loss_scaler.scale(loss)

        return loss, logits
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(input_ids, attention_mask, token_type_ids, label):
        status = init_register()
        input_ids = ops.depend(input_ids, status)
        (loss, _), grads = grad_fn(input_ids, attention_mask, token_type_ids, label)
        # 混合精度需要
        status = all_finite(grads, status)
        if status:
            loss = loss_scaler.unscale(loss)
            grads = loss_scaler.unscale(grads)
            loss = ops.depend(loss, optimizer(grads))
        loss = ops.depend(loss, loss_scaler.adjust(status))

        return loss, status

    size = train_dataset.get_dataset_size()
    # 修改loss
    loss_total = 0
    batch_size = train_dataset.get_batch_size()
    model.set_train(True)

    with tqdm(total=size * batch_size) as t:
        for batch, data in enumerate(train_dataset.create_dict_iterator()):
            t.set_description('Epoch %i' % current_epoch)
            loss, status = train_step(data['input_ids'], data['attention_mask'], data['token_type_ids'], data['label'])
            status = status.asnumpy()
            if status:
                loss_total = loss_total + loss.asnumpy()
            else:
                print(f"grads overflow, skip step")
        
            loss, current = loss.asnumpy(), batch
            t.set_postfix(loss=loss_total/current)
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
    mindspore.set_context(device_target="Ascend")
    save_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    args = init_sst2_args()
    try:
        import zipfile
        f = zipfile.ZipFile("{}-{}.zip".format(args.dataset_path, args.max_length))
        f.extractall()
        f.close()
    except Exception:
        raise Exception("解压数据集出错")

    best_accuracy = 0
    context.set_context(mode=context.GRAPH_MODE)
    # context.set_context(mode=context.PYNATIVE_MODE)
    # train set
    
    epoch_num = args.epochs
    ckpt_file = args.bert_ckpt
    config = BertConfig(args.config)
    bert_sst_2 = BertBinaryClassificationModel(config, ckpt_file)

    # use amp
    from src.amp import all_finite, auto_mixed_precision, DynamicLossScaler, NoLossScaler, init_register
    if args.amp:
        model = auto_mixed_precision(bert_sst_2, 'O1')
        loss_scaler = DynamicLossScaler(1024, 2, 1000)
    else:
        loss_scaler = NoLossScaler()

    params = bert_sst_2.trainable_params()
    # get datset
    train_dataset ,test_dataset = get_sst2_dataset(args.dataset_path, args.train_batch_size, args.test_batch_size, args.max_length)
    lr_schedule = BertLearningRate(learning_rate=args.lr,
                                   end_learning_rate=0.0,
                                   warmup_steps=int(train_dataset.get_dataset_size() * epoch_num * 0.1),
                                   decay_steps=train_dataset.get_dataset_size() * epoch_num,
                                   power=1.0)
    optimizer = nn.AdamWeightDecay(params, lr_schedule, eps=1e-8)
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_and_test(bert_sst_2, loss_scaler, train_dataset, test_dataset, optimizer, epoch+1)
    print("Done! Best Accuracy is:", best_accuracy)
    # only test for fine-tuning
    # check_save_sst2_ckpt(test_dataset, config,'outputs/4_test_4_epoch_1000_step_acc_82.91284403669725%.ckpt')
