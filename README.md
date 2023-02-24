# BERT
<!-- TOC -->
- [目录](#目录)
- [BERT概述](#Bert概述)
- [项目简述](#项目概述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
    - [脚本和样例代码](#脚本和样例代码)
    - [脚本参数](#脚本参数)
        - [预训练](#预训练)
        - [微调与评估](#微调与评估)
    - [选项及参数](#选项及参数)
        - [选项](#选项)
        - [参数](#参数)
    - [训练过程](#训练过程)
        - [用法](#用法)
            - [Ascend处理器上运行](#ascend处理器上运行)
        - [分布式训练](#分布式训练)
            - [Ascend处理器上运行](#ascend处理器上运行-1)
    - [评估过程](#评估过程)
        - [用法](#用法-1)
            - [Ascend处理器上运行后评估sst2数据集](#ascend处理器上运行后评估cola数据集)

<!-- /TOC -->

# BERT概述

BERT网络由谷歌在2018年提出，该网络在自然语言处理领域取得了突破性进展。采用预训练技术，实现大的网络结构，并且仅通过增加输出层，实现多个基于文本的任务的微调。BERT的主干代码采用Transformer的Encoder结构。引入注意力机制，使输出层能够捕获高纬度的全局语义信息。预训练采用去噪和自编码任务，即掩码语言模型（MLM）和相邻句子判断（NSP）。无需标注数据，可对海量文本数据进行预训练，仅需少量数据做微调的下游任务，可获得良好效果。BERT所建立的预训练加微调的模式在后续的NLP网络中得到了广泛的应用。

[论文](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.[BERT：深度双向Transformer语言理解预训练](https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

如果您对小规格Bert预训练的蒸馏、剪枝感兴趣可以参考论文[Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962)。这能使小规格的Bert性能更优，目前[cqunlp/bert](https://github.com/cqunlp/bert/tree/master)项目暂时未采用上诉方法。

# 项目简述
本项目专注于Bert较小的规格，目前支持Bert-L4-H128、Bert-L4-H256、Bert-L4-H512、Bert-L4-H768、Bert-L6-H256、Bert-L6-H512、Bert-L6-H768、Bert-L8-H512、Bert-L6-H768规格的Bert，详见[Bert官方仓库](https://github.com/google-research/bert)
# 模型架构

BERT的主干结构为Transformer。BERT由两个组成部分构成：Transformer编码器和预训练任务。Transformer编码器是BERT的主体，由多个Transformer块组成，每个块包含了一个自注意力机制和一些全连接层。预训练任务是在大量未标注数据上训练出来的，包括掩码语言模型和下一句预测任务。

具体来说，BERT的输入是一组token，每个token由三个部分组成：token本身、segment ID和position embedding。其中，segment ID用于区分两个句子，position embedding表示每个token在输入序列中的位置。BERT将输入序列通过一系列的Transformer编码器，每个编码器都包含一个自注意力机制和一个前馈神经网络（feedforward neural network）。自注意力机制可以对输入序列中的所有token进行建模，然后得到一个表示整个序列的向量。这些向量在不同的编码器之间不断传递和更新，最终得到一个整个序列的表示，称为BERT的输出。

# 数据集

- 生成预训练数据集
    - 下载[enwiki](https://dumps.wikimedia.org/enwiki/)数据集进行预训练，
    - 使用[WikiExtractor](https://github.com/attardi/wikiextractor)提取和整理数据集中的文本，使用步骤如下：
        - pip install wikiextractor
        - python -m wikiextractor.WikiExtractor -o <output file path> -b <output file size> <Wikipedia dump file>
    - `WikiExtarctor`提取出来的原始文本并不能直接使用，还需要将数据集预处理并转换为TFRecord格式。详见[BERT](https://github.com/google-research/bert#pre-training-with-bert)代码仓中的create_pretraining_data.py文件，同时下载对应的vocab.txt文件, 如果出现AttributeError: module 'tokenization' has no attribute 'FullTokenizer’，请安装bert-tensorflow。



# 环境要求

- 硬件（Ascend处理器或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.8/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/r1.8/index.html)
- 软件环境，MindSpore：1.8.1

# 快速入门

从官网下载安装MindSpore之后，您可以按照如下步骤进行训练和评估：

- 在Ascend上运行
```bash
# 单卡运行测试示例
#切换到ascend分支
git checkout ascend
python run_pretrain_test_one_card.py

# 多卡并行运行预训练示例
#切换到ascend分支
git checkout ascend
bash run_pretrain_ascend.sh

# 运行微调sst-2和评估示例

- 运行`bash run_classify_sst2.bash`，对预训练完成的BERT模型进行微调。
  bash run_classify_sst2.bash
```

- 在GPU上运行

```bash

# 单机运行测试示例
#切换到bert-mini分支
git checkout bert-mini
python run_pretrain_test_one_card.py

# 分布式运行预训练示例
#切换到主分支
git checkout main
bash run_pretrain_bert.sh

# 运行微调sst-2和评估示例

- 运行`bash run_classify_sst2.bash`，对预训练完成的BERT模型进行微调。
  bash run_classify_sst2.bash
```

## 脚本说明

## 脚本和样例代码

```shell
.
└─bert
  ├─README.md
  ├─finetuningSST2
    ├─outputs                                 # 精度达标保存目录
    ├─result_log                              # 微调log目录
    ├─SST2                                    # sst2原始数据集
    ├─model.py                                # bert下游任务模型
    ├─run_classify.py                         # 微调、模型精度测试脚本
    ├─run_classify.ipynb                      # 微调、模型测试notebook
  ├─config                                    # bert模型json配置目录
  ├─outputs                                   # 模型预训练结果
  ├─src
    ├─amp.py                                  # 混合精度代码
    ├─api.py                                  # MindSpore自动微分求导
    ├─bert.py                                 # bert模型骨干结构
    ├─config.py                               # bert模型config定义
    ├─metric.py                               # 评价指标
    ├─optimization.py                         # bert优化器
    ├─tokenizer.py                            # bert 分词器
    ├─utils.py                                # 预训练功能函数
  ├─vocab                                     # bert词表文件
  ├─run_classify_sst2.sh                      # 下游任务shell脚本
  ├─run_pretrain_bert.sh                      # bert预训练shell脚本
  ├─run_pretrain.py                           # bert预训练脚本
```

## 脚本参数

### 预训练

```shell
用法：run_pretrain.py  
                        [--data_dir] 
                        [--use_ascend] 
                        [--jit] 
                        [--do_train] 
                        [--lr] 
                        [--warmup_stepsT]
                        [--train_batch_size]
                        [--epochs] 
                        [--do_save_ckpt]
                        [--save_steps]
                        [--save_ckpt_path]
                        [--do_load_ckpt]
                        [--model_path] 
                        [--config]

选项：
    [--data_dir]             # 预训练mindrecord数据路径
    [--use_ascend]           # 是否在ascend上运行
    [--jit]                  # 是否采用jit
    [--do_train]             # 是否预训练
    [--lr]                   # 学习率大小，默认为2e-5
    [--warmup_steps]         # warmup steps数，默认为1000
    [--train_batch_size]     # train batch size 大小，默认为128
    [--epochs]               # epoch数，默认为15
    [--do_save_ckpt]         # 是否保存预训练ckpt
    [--save_steps]           # 保存ckpt间隔步数
    [--save_ckpt_path]       # 保存的ckpt路径
    [--do_load_ckpt]         # 是否预先载入模型ckpt，一般用于两阶段预训练，默认为False
    [--model_path]           # 若要预先载入模型ckpt，其ckpt路径位置
    [--config]               # 该规格Bert对应的ckpt
```

### 微调与评估

```shell
usage: run_classifier.py 
[--device_target]
[--amp]
[--bert_ckpt]
[--dataset_path]
[--config]
[--train_batch_size]
[--test_batch_size]
[--epochs]
[--lr]
[--max_length]
[--acc]
options:
    --device_target                   任务运行的目标设备，可选项为Ascend或GPU
    --amp                             是否采用混合精度训练，可选项为true或false
    --bert_ckpt                       需要微调的bert的ckpt路径
    --dataset_path                    数据集路径，默认为sst2
    --config                          该规格bert的config json文件路径
    --train_batch_size                训练batch size大小
    --test_batch_size                 测试batch size大小
    --epochs                          训练epoch轮数
    --lr                              学习率大小，默认为2e-5
    --max_length                      对文本进行padding或者截断的长度，默认为64，可选64、128、256、512
    --load_pretrain_checkpoint_path   初始检查点（通常来自预训练BERT模型）
    --acc                             测试保存精度
```

### 参数

```text
数据集和网络参数（预训练/微调/评估）：
    seq_length                      输入序列的长度，默认为config文件传入数值
    vocab_size                      各内嵌向量大小，需与所采用的数据集相同，默认为config文件传入数值
    hidden_size                     BERT的encoder隐藏维度数，默认为config文件传入数值
    num_hidden_layers               隐藏层数，默认为config文件传入数值
    num_attention_heads             注意头的数量，默认为config文件传入数值
    intermediate_size               中间层数，默认为config文件传入数值
    hidden_act                      所采用的激活函数，默认为gelu
    hidden_dropout_prob             BERT输出的随机失活可能性，默认为0.1
    attention_probs_dropout_prob    BERT注意的随机失活可能性，默认为0.1
    max_position_embeddings         序列最大长度，默认为512
    type_vocab_size                 标记类型的词汇表大小，默认为16
    initializer_range               TruncatedNormal的初始值，默认为0.02
    use_relative_positions          是否采用相对位置，可选项为true或false，默认为False
    dtype                           输入的数据类型，可选项为mstype.float16或mstype.float32，默认为mstype.float32
    compute_type                    Bert Transformer的计算类型，可选项为mstype.float16或mstype.float32，默认为mstype.float16

Parameters for optimizer:
    AdamWeightDecay:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率，取值需为正数
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
    eps                             增加分母，提高小数稳定性

    Lamb:
    decay_steps                     学习率开始衰减的步数
    learning_rate                   学习率
    end_learning_rate               结束学习率
    power                           幂
    warmup_steps                    热身学习率步数
    weight_decay                    权重衰减
```

## 训练过程

### 用法

#### Ascend处理器上运行

```bash
bash run_pretrain_ascend.sh
```

以上命令后台运行，您可以在/outputs中查看训练日志。训练结束后，您可以得到如下损失值：

```text

avg_time(ms): 402.258091, rank_id: 7, cur_step: 189800, skip_steps:   0, train_step: 189800, loss: 0.411356, masked_lm_loss: 0.145585, next_sentence_loss: 0.265771
```

## 评估过程

### 用法

#### Ascend处理器上运行后评估sst2数据集

运行以下命令前，确保已设置对应Bert规格的config文件，预训练后的Bert checkpoint ckpt文件，例如，

--bert_ckpt="/data/checkpoint/checkpoint_bert_L4_H768.ckpt" \

--config="/data/bert/config.json"

```bash
bash run_classify_sst2.bash
```

以上命令后台运行，您可以在result_log/finetuningsst2_${time}.log中查看训练日志。

如您选择准确性作为评估方法，可得到如下结果：

```text
Test: 
 Accuracy: 90.3%, Best-Accuracy: 91.1%, Avg loss: 0.340414 
```
