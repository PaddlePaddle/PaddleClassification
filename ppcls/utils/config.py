# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import argparse
import yaml
from . import logger
from . import check
from collections import OrderedDict

__all__ = ['get_config', 'convert_to_dict']


def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    else:
        return obj


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return AttrDict(copy.deepcopy(dict(self)))


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in d.items():
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", k))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", k))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))

        if k[0].isupper() and delimiter == 0:
            logger.info(placeholder)


def print_config(config):
    """
    visualize configs
    Arguments:
        config: configs
    """
    logger.advertise()
    print_dict(config)


def check_config(config):
    """
    Check config
    """
    check.check_version()
    use_gpu = config.get('use_gpu', True)
    if use_gpu:
        check.check_gpu()
    architecture = config.get('ARCHITECTURE')
    #check.check_architecture(architecture)
    use_mix = config.get('use_mix', False)
    check.check_mix(architecture, use_mix)
    classes_num = config.get('classes_num')
    check.check_classes_num(classes_num)
    mode = config.get('mode', 'train')
    if mode.lower() == 'train':
        check.check_function_params(config, 'LEARNING_RATE')
        check.check_function_params(config, 'OPTIMIZER')


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
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
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
                print('A new field ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            if ks[0] not in dl.keys():
                dl[ks[0]] = {}
                print("A new Series field ({}) detected!".format(ks[0], dl))
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
            assert isinstance(opt, str), (
                "option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)
    return config


def get_config(fname, overrides=None, show=False):
    """
    Read config from file
    """
    assert os.path.exists(fname), ('config file({}) is not exist'.format(fname))
    config = parse_config(fname)
    override_config(config, overrides)
    if show:
        print_config(config)
    # check_config(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser("generic-image-rec train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    args = parser.parse_args()
    return args


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_orderdict():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def dump_infer_config(inference_config, path):
    setup_orderdict()
    infer_cfg = OrderedDict()
    config = copy.deepcopy(inference_config)
    if config["Global"].get("pdx_model_name", None):
        infer_cfg["Global"] = {"model_name": config["Global"]["pdx_model_name"]}
    if config.get("Infer"):
        transforms = config["Infer"]["transforms"]
    elif config["DataLoader"]["Eval"].get("Query"):
        transforms = config["DataLoader"]["Eval"]["Query"]["dataset"][
            "transform_ops"]
        transforms.append({"ToCHWImage": None})
    elif config["DataLoader"]["Eval"].get("dataset"):
        transforms = config["DataLoader"]["Eval"]["dataset"]["transform_ops"]
        transforms.append({"ToCHWImage": None})
    else:
        logger.error("This config does not support dump transform config!")
    transform = next((item for item in transforms if 'CropImage' in item), None)
    if transform:
        dynamic_shapes = transform["CropImage"]["size"]
    else:
        transform = next((item for item in transforms
                          if 'ResizeImage' in item), None)
        if transform:
            if isinstance(transform["ResizeImage"]["size"], list):
                dynamic_shapes = transform["ResizeImage"]["size"][0]
            elif isinstance(transform["ResizeImage"]["size"], int):
                dynamic_shapes = transform["ResizeImage"]["size"]
            else:
                raise ValueError(
                    "ResizeImage size must be either a list or an int.")
        else:
            raise ValueError("No valid transform found.")
    # Configuration required config for high-performance inference.
    if config["Global"].get("hpi_config_path", None):
        hpi_config = convert_to_dict(
            parse_config(config["Global"]["hpi_config_path"]))
        if hpi_config["Hpi"]["backend_config"].get("paddle_tensorrt", None):
            hpi_config["Hpi"]["backend_config"]["paddle_tensorrt"][
                "dynamic_shapes"]["x"] = [[
                    1, 3, dynamic_shapes, dynamic_shapes
                ] for i in range(3)]
            hpi_config["Hpi"]["backend_config"]["paddle_tensorrt"][
                "max_batch_size"] = 1
        if hpi_config["Hpi"]["backend_config"].get("tensorrt", None):
            hpi_config["Hpi"]["backend_config"]["tensorrt"]["dynamic_shapes"][
                "x"] = [[1, 3, dynamic_shapes, dynamic_shapes]
                        for i in range(3)]
            hpi_config["Hpi"]["backend_config"]["tensorrt"][
                "max_batch_size"] = 1
        infer_cfg["Hpi"] = hpi_config["Hpi"]
    for transform in transforms:
        if "NormalizeImage" in transform:
            transform["NormalizeImage"]["channel_num"] = 3
            scale_str = transform["NormalizeImage"]["scale"]
            numerator, denominator = scale_str.split('/')
            numerator, denominator = float(numerator), float(denominator)
            transform["NormalizeImage"]["scale"] = float(numerator /
                                                         denominator)
    infer_cfg["PreProcess"] = {
        "transform_ops": [
            infer_preprocess for infer_preprocess in transforms
            if "DecodeImage" not in infer_preprocess
        ]
    }
    if config.get("Infer"):
        postprocess_dict = config["Infer"]["PostProcess"]

        with open(postprocess_dict["class_id_map_file"], 'r') as f:
            label_id_maps = f.readlines()
        label_names = []
        for line in label_id_maps:
            line = line.strip().split(' ', 1)
            label_names.append(line[1:][0])

        postprocess_name = postprocess_dict.get("name", None)
        postprocess_dict.pop("class_id_map_file")
        postprocess_dict.pop("name")
        dic = OrderedDict()
        for item in postprocess_dict.items():
            dic[item[0]] = item[1]
        dic['label_list'] = label_names

        if postprocess_name:
            infer_cfg["PostProcess"] = {postprocess_name: dic}
        else:
            raise ValueError("PostProcess name is not specified")
    else:
        infer_cfg["PostProcess"] = {"NormalizeFeatures": None}
    with open(path, 'w') as f:
        yaml.dump(infer_cfg, f)
    logger.info("Export inference config file to {}".format(os.path.join(path)))
