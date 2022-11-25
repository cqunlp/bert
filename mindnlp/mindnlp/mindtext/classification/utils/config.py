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
Parse configure from yaml.
"""

import argparse
import os
import logging
import yaml


def parse_args():
    """
    Parse arguments from `yaml` config file.

    Returns:
        object: argparse object.
    """
    parser = argparse.ArgumentParser("MindText cls train script.")
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default="",
                        help='fasttext config file path.')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden.')
    return parser.parse_args()


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def create_attr_dict(yaml_config):
    """

    Args:
        yaml_config:

    Returns:

    """
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:  # pylint: disable=unidiomatic-typecheck
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:  # pylint: disable=broad-except
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def override(dl, ks, v):
    """
    Recursively replace dict of list

    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """

    def str2num(v):
        try:
            return eval(v)  # pylint: disable=eval-used
        except Exception:  # pylint: disable=broad-except
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")  # pylint: disable=no-member
    assert len(ks) > 0, ('lenght of keys should larger than 0')  # pylint: disable=len-as-condition
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                logging.warning \
                    ('A new filed ({}) detected!'.format(ks[0]))  # pylint: disable=logging-format-interpolation
            dl[ks[0]] = str2num(v)
        else:
            if not ks[0] in dl:
                logging.warning \
                    ('A new filed ({}) detected!'.format(ks[0]))  # pylint: disable=logging-format-interpolation
                dl[ks[0]] = {}
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config

    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]

    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt,
                              str), ("option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)

    return config


def get_config(fname, overrides=None):
    """
    Read config from file
    """
    assert os.path.exists(fname), (
        'config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    override_config(config, overrides)
    return config


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config
