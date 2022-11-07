# 快速入门

此教程主要针对初级用户，内容主要包括利用`MindText`的`classification`进行网络训练。

## 目录结构说明

```text
mindtext/classification/
├── config    # 模型、数据集等配置文件目录
│   └── fasttext
│       └── fasttext.yaml   # 模型配置文件
├── dataset   # 数据集的读取和处理
│   ├── dataset.py                    # 生成训练、验证和测试数据
│   ├── FastTextDataPreProcess.py     # 数据预处理
│   ├── load_data.py                  # 加载数据
│   └── mindrecord.py                 # 中间数据文件处理
├── docs
│   └── getting_started.py # 用户使用指南
├── model
│   ├── backbones
│   |    └── fasttext.py  # 模型的骨干架构
|   ├── classifiers
│   |    └── base.py      # 模型分类器
|   ├── build_model.py    # 创建模型
│   ├── loss.py           # 创建loss
│   └── optimizer.py      # 创建优化器
├── test    # 测试模型  
├── tools
│   ├── eval.py     # 评估模型
│   ├── export.py   # 导出模型的checkpoint文件
│   ├── infer.py    # 模型推理
│   └── train.py    # 训练模型
└── utils
    ├── config.py        # 处理yaml文件
    └── lr_schedule.py   # 学习率设置（可选）
```

## 环境安装与配置

下载`MindText`并进入文件夹

```shell
git clone https://gitee.com/mindspore/mindtext.git
cd mindtext
```

安装

```shell
python setup.py install
```

所需环境：

```bash
pip install pandas==1.2.4
pip install numpy==1.20.3
pip install mindspore==1.2.0
pip install tqdm==4.61.1
pip install PyYAML==5.4.1
pip install scikit_learn==0.24.1
pip install spacy==2.3.1
python -m spacy download en_core_web_lg==2.3.1
```

## 数据准备

下载并解压数据集.

你可以从数据集下载页面下载，并按下方目录结构放置：

```text
/root/fasttext/ag_news_csv
├── ag_news_csv
│   ├── train.csv
│   └── text.csv
├── dbpedia_csv
│   ├── train.csv
│   └── text.csv
├── yelp_review_polarity_csv
│   ├── train.csv
│   └── text.csv

```

## 自定义配置文件

进入`./config/fasttext`目录，打开`fasttext.yaml`文件

`fasttext.yaml`文件中有多个参数配置, 案例如下：

```text
# Builtin Configurations

model_name: "fasttext"      # 模型名称
device_target: "GPU"        # 设备，可选GPU, ASCEND

PREPROCESS:                                     # 数据预处理参数
  max_len: 467                                  # 数据集最长文本长度
  mid_dir_path: "./ag_temp_data"                # mindrecord生成路径
  vocab_file_path: "your_path/vocab.txt"        # 生成/读取词表路径

MODEL_PARAMETERS:           # 模型参数
  vocab_size: 1383812       # 词表大小
  embedding_dims: 16        # 词嵌入大小
  num_class: 4              # 类别数

OPTIMIZER:                  # 优化器参数
  function: "Adam"          # 优化器类型，以Adam优化器为例
  lr: 0.20                  # 学习率
  min_lr: 0.000001          # 最小学习率
  decay_steps: 236          # 学习率衰减补偿
  warmup_steps: 400000                # warm_up步长
  poly_lr_scheduler_power: 0.001      # 学习率策略

TRAIN:                                      # 训练参数
  data_path: "your_path/train.csv"          # 训练集路径
  batch_size: 512                           # batch_size
  buckets: [64, 128, 467]                   # 训练集数据加载块大小
  epoch: 5                                  # 训练epoch数
  epoch_count: 1
  loss_function: "SoftmaxCrossEntropyWithLogits"    # 损失函数类型
  pretrain_ckpt_dir: ""                             # 断点训练检查点
  save_ckpt_steps: 116                              # 检查点保存步长
  save_ckpt_dir: "your_path"                        # 检查点保存路径
  keep_ckpt_max: 10                                 # 最大检查点数
  run_distribute: False                             # 分布式训练，默认False
  distribute_batch_size_gpu: 64                     # 分布式训练单卡batch_size

VALID:                                          # 测试参数
  data_path: "your_path/test.csv"               # 测试集路径
  batch_size: 512                               # batch_size
  model_ckpt: "your_path/fasttext-*_***.ckpt"   # 模型检查点
  test_buckets: [467]                           # 测试集数据加载块大小

INFER:                                          # 推断参数
  data_path: "your_path/test.csv"               # 推断数据路径
  batch_size: 2048                              # batch_size
  model_ckpt: "your_path/fasttext-*_***.ckpt"   # 模型检查点
  buckets: [467]                                # 推断数据加载块大小

EXPORT:                                         # 模型导出参数
  device_id: 0                                  # 设备id
  ckpt_file: "your_path/fasttext-*_***.ckpt"    # 检查点路径
  file_name: "fasttexts"                        # 文件名称
  file_format: "AIR"                            # 文件类型，可选AIR, ONNX, MINDIR
```

## 模型训练

进入`mindtext/classification/tools`目录。

```bash
cd mindtext/classification/tools
```

执行下面的命令开始模型训练：

```shell
python train.py -c ../configs/fasttext/fasttext.yaml
```

## 模型评估

进入`mindtext/classification/tools`目录。

执行下面的命令开始模型评估：

```shell
python eval.py -c ../configs/fasttext/fasttext.yaml
```

## 模型导出

进入`mindtext/classification/tools`目录。

执行下面的命令开始模型导出：

```shell
python export.py -c ../configs/fasttext/fasttext.yaml
```

## 模型预测

进入`mindtext/classification/tools`目录。

执行下面的命令开始模型预测：

```shell
python infer.py -c ../configs/fasttext/fasttext.yaml
```

- `-c` 参数是指定训练的配置文件路径，训练的具体超参数可查看`yaml`文件  