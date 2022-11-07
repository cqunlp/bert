# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Mindrecord file
"""
import os
from mindspore.mindrecord import FileWriter
import numpy as np


def mindrecord_file_path(config, temp_dir, data_example):
    file_name = os.path.splitext(os.path.basename(config.data_path))[0]
    if os.path.exists(os.path.join(os.getcwd(), temp_dir)):
        print("MindRecord data already exist")
        data_path = os.path.join(os.getcwd(), temp_dir, file_name)
    else:
        print("Writing data to MindRecord file......")
        dir_path = os.path.join(os.getcwd(), temp_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        data_path = os.path.join(dir_path, file_name)
        for i in config.buckets:
            write_to_mindrecord(data_example[i], data_path + '_' + str(i) + '.mindrecord', 1)
    return data_path


def write_to_mindrecord(data, path, shared_num=1):
    """generate mindrecord"""
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    writer = FileWriter(path, shared_num)
    data_schema = {
        "src_tokens": {"type": "int32", "shape": [-1]},
        "src_tokens_length": {"type": "int32", "shape": [-1]},
        "label_idx": {"type": "int32", "shape": [-1]}
    }
    writer.add_schema(data_schema, "fasttext")
    for item in data:
        item['src_tokens'] = np.array(item['src_tokens'], dtype=np.int32)
        item['src_tokens_length'] = np.array(item['src_tokens_length'], dtype=np.int32)
        item['label_idx'] = np.array(item['label_idx'], dtype=np.int32)
        writer.write_raw_data([item])
    writer.commit()
