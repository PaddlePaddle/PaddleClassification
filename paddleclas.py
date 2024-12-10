# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Union, Generator
import argparse
import shutil
import textwrap
import tarfile
import requests
from functools import partial
from difflib import SequenceMatcher

import cv2
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import paddle

from .ppcls.arch import backbone
from .ppcls.utils import logger

from .deploy.python.predict_cls import ClsPredictor
from .deploy.python.predict_system import SystemPredictor
from .deploy.python.build_gallery import GalleryBuilder
from .deploy.utils.get_image_list import get_image_list
from .deploy.utils import config

# for the PaddleClas Project
from . import deploy
from . import ppcls

# for building model with loading pretrained weights from backbone
logger.init_logger()

__all__ = ["PaddleClas"]

BASE_DIR = os.path.expanduser(os.path.join("~", ".paddleclas"))
BASE_INFERENCE_MODEL_DIR = os.path.join(BASE_DIR, "inference_model")
BASE_IMAGES_DIR = os.path.join(BASE_DIR, "images")
IMN_MODEL_BASE_DOWNLOAD_URL = "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/{}_infer.tar"
IMN_MODEL_SERIES = {
    "AlexNet": ["AlexNet"],
    "ConvNeXt": ["ConvNeXt_tiny"],
    "CSPNet": ["CSPDarkNet53"],
    "CSWinTransformer": [
        "CSWinTransformer_tiny_224", "CSWinTransformer_small_224",
        "CSWinTransformer_base_224", "CSWinTransformer_base_384",
        "CSWinTransformer_large_224", "CSWinTransformer_large_384"
    ],
    "DarkNet": ["DarkNet53"],
    "DeiT": [
        "DeiT_base_distilled_patch16_224", "DeiT_base_distilled_patch16_384",
        "DeiT_base_patch16_224", "DeiT_base_patch16_384",
        "DeiT_small_distilled_patch16_224", "DeiT_small_patch16_224",
        "DeiT_tiny_distilled_patch16_224", "DeiT_tiny_patch16_224"
    ],
    "DenseNet": [
        "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
        "DenseNet264"
    ],
    "DLA": [
        "DLA46_c", "DLA60x_c", "DLA34", "DLA60", "DLA60x", "DLA102", "DLA102x",
        "DLA102x2", "DLA169"
    ],
    "DPN": ["DPN68", "DPN92", "DPN98", "DPN107", "DPN131"],
    "EfficientNet": [
        "EfficientNetB0", "EfficientNetB0_small", "EfficientNetB1",
        "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5",
        "EfficientNetB6", "EfficientNetB7"
    ],
    "ESNet": ["ESNet_x0_25", "ESNet_x0_5", "ESNet_x0_75", "ESNet_x1_0"],
    "GhostNet":
    ["GhostNet_x0_5", "GhostNet_x1_0", "GhostNet_x1_3", "GhostNet_x1_3_ssld"],
    "HarDNet": ["HarDNet39_ds", "HarDNet68_ds", "HarDNet68", "HarDNet85"],
    "HRNet": [
        "HRNet_W18_C", "HRNet_W30_C", "HRNet_W32_C", "HRNet_W40_C",
        "HRNet_W44_C", "HRNet_W48_C", "HRNet_W64_C", "HRNet_W18_C_ssld",
        "HRNet_W48_C_ssld"
    ],
    "Inception": ["GoogLeNet", "InceptionV3", "InceptionV4"],
    "LeViT":
    ["LeViT_128S", "LeViT_128", "LeViT_192", "LeViT_256", "LeViT_384"],
    "MixNet": ["MixNet_S", "MixNet_M", "MixNet_L"],
    "MobileNetV1": [
        "MobileNetV1_x0_25", "MobileNetV1_x0_5", "MobileNetV1_x0_75",
        "MobileNetV1", "MobileNetV1_ssld"
    ],
    "MobileNetV2": [
        "MobileNetV2_x0_25", "MobileNetV2_x0_5", "MobileNetV2_x0_75",
        "MobileNetV2", "MobileNetV2_x1_5", "MobileNetV2_x2_0",
        "MobileNetV2_ssld"
    ],
    "MobileNetV3": [
        "MobileNetV3_small_x0_35", "MobileNetV3_small_x0_5",
        "MobileNetV3_small_x0_75", "MobileNetV3_small_x1_0",
        "MobileNetV3_small_x1_25", "MobileNetV3_large_x0_35",
        "MobileNetV3_large_x0_5", "MobileNetV3_large_x0_75",
        "MobileNetV3_large_x1_0", "MobileNetV3_large_x1_25",
        "MobileNetV3_small_x1_0_ssld", "MobileNetV3_large_x1_0_ssld"
    ],
    "MobileViT": ["MobileViT_XXS", "MobileViT_XS", "MobileViT_S"],
    "PeleeNet": ["PeleeNet"],
    "PPHGNet": [
        "PPHGNet_tiny",
        "PPHGNet_small",
        "PPHGNet_tiny_ssld",
        "PPHGNet_small_ssld",
    ],
    "PPLCNet": [
        "PPLCNet_x0_25", "PPLCNet_x0_35", "PPLCNet_x0_5", "PPLCNet_x0_75",
        "PPLCNet_x1_0", "PPLCNet_x1_5", "PPLCNet_x2_0", "PPLCNet_x2_5"
    ],
    "PPLCNetV2": ["PPLCNetV2_base"],
    "PVTV2": [
        "PVT_V2_B0", "PVT_V2_B1", "PVT_V2_B2", "PVT_V2_B2_Linear", "PVT_V2_B3",
        "PVT_V2_B4", "PVT_V2_B5"
    ],
    "RedNet": ["RedNet26", "RedNet38", "RedNet50", "RedNet101", "RedNet152"],
    "RegNet": ["RegNetX_4GF"],
    "Res2Net": [
        "Res2Net50_14w_8s", "Res2Net50_26w_4s", "Res2Net50_vd_26w_4s",
        "Res2Net200_vd_26w_4s", "Res2Net101_vd_26w_4s",
        "Res2Net50_vd_26w_4s_ssld", "Res2Net101_vd_26w_4s_ssld",
        "Res2Net200_vd_26w_4s_ssld"
    ],
    "ResNeSt": ["ResNeSt50", "ResNeSt50_fast_1s1x64d"],
    "ResNet": [
        "ResNet18", "ResNet18_vd", "ResNet34", "ResNet34_vd", "ResNet50",
        "ResNet50_vc", "ResNet50_vd", "ResNet50_vd_v2", "ResNet101",
        "ResNet101_vd", "ResNet152", "ResNet152_vd", "ResNet200_vd",
        "ResNet34_vd_ssld", "ResNet50_vd_ssld", "ResNet50_vd_ssld_v2",
        "ResNet101_vd_ssld", "Fix_ResNet50_vd_ssld_v2", "ResNet50_ACNet_deploy"
    ],
    "ResNeXt": [
        "ResNeXt50_32x4d", "ResNeXt50_vd_32x4d", "ResNeXt50_64x4d",
        "ResNeXt50_vd_64x4d", "ResNeXt101_32x4d", "ResNeXt101_vd_32x4d",
        "ResNeXt101_32x8d_wsl", "ResNeXt101_32x16d_wsl",
        "ResNeXt101_32x32d_wsl", "ResNeXt101_32x48d_wsl",
        "Fix_ResNeXt101_32x48d_wsl", "ResNeXt101_64x4d", "ResNeXt101_vd_64x4d",
        "ResNeXt152_32x4d", "ResNeXt152_vd_32x4d", "ResNeXt152_64x4d",
        "ResNeXt152_vd_64x4d"
    ],
    "ReXNet":
    ["ReXNet_1_0", "ReXNet_1_3", "ReXNet_1_5", "ReXNet_2_0", "ReXNet_3_0"],
    "SENet": [
        "SENet154_vd", "SE_HRNet_W64_C_ssld", "SE_ResNet18_vd",
        "SE_ResNet34_vd", "SE_ResNet50_vd", "SE_ResNeXt50_32x4d",
        "SE_ResNeXt50_vd_32x4d", "SE_ResNeXt101_32x4d"
    ],
    "ShuffleNetV2": [
        "ShuffleNetV2_swish", "ShuffleNetV2_x0_25", "ShuffleNetV2_x0_33",
        "ShuffleNetV2_x0_5", "ShuffleNetV2_x1_0", "ShuffleNetV2_x1_5",
        "ShuffleNetV2_x2_0"
    ],
    "SqueezeNet": ["SqueezeNet1_0", "SqueezeNet1_1"],
    "SwinTransformer": [
        "SwinTransformer_large_patch4_window7_224_22kto1k",
        "SwinTransformer_large_patch4_window12_384_22kto1k",
        "SwinTransformer_base_patch4_window7_224_22kto1k",
        "SwinTransformer_base_patch4_window12_384_22kto1k",
        "SwinTransformer_base_patch4_window12_384",
        "SwinTransformer_base_patch4_window7_224",
        "SwinTransformer_small_patch4_window7_224",
        "SwinTransformer_tiny_patch4_window7_224"
    ],
    "Twins": [
        "pcpvt_small", "pcpvt_base", "pcpvt_large", "alt_gvt_small",
        "alt_gvt_base", "alt_gvt_large"
    ],
    "TNT": ["TNT_small"],
    "VAN": ["VAN_B0"],
    "VGG": ["VGG11", "VGG13", "VGG16", "VGG19"],
    "VisionTransformer": [
        "ViT_base_patch16_224", "ViT_base_patch16_384", "ViT_base_patch32_384",
        "ViT_large_patch16_224", "ViT_large_patch16_384",
        "ViT_large_patch32_384", "ViT_small_patch16_224"
    ],
    "Xception": [
        "Xception41", "Xception41_deeplab", "Xception65", "Xception65_deeplab",
        "Xception71"
    ]
}

PULC_MODEL_BASE_DOWNLOAD_URL = "https://paddleclas.bj.bcebos.com/models/PULC/inference/{}_infer.tar"
PULC_MODELS = [
    "car_exists", "language_classification", "person_attribute",
    "person_exists", "safety_helmet", "text_image_orientation",
    "image_orientation", "textline_orientation", "traffic_sign",
    "vehicle_attribute", "table_attribute", "clarity_assessment"
]

SHITU_MODEL_BASE_DOWNLOAD_URL = "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/{}_infer.tar"
SHITU_MODELS = [
    # "picodet_PPLCNet_x2_5_mainbody_lite_v1.0",  # ShiTuV1(V2)_mainbody_det
    # "general_PPLCNet_x2_5_lite_v1.0"  # ShiTuV1_general_rec
    # "PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0",  # ShiTuV2_general_rec TODO(hesensen): add lite model
    "PP-ShiTuV2"
]


class ImageTypeError(Exception):
    """ImageTypeError.
    """

    def __init__(self, message=""):
        super().__init__(message)


class InputModelError(Exception):
    """InputModelError.
    """

    def __init__(self, message=""):
        super().__init__(message)


def init_config(model_type, model_name, inference_model_dir, **kwargs):

    if kwargs.get("build_gallery", False):
        cfg_path = "deploy/configs/inference_general.yaml"
    elif model_type == "pulc":
        cfg_path = f"deploy/configs/PULC/{model_name}/inference_{model_name}.yaml"
    elif model_type == "shitu":
        cfg_path = "deploy/configs/inference_general.yaml"
    else:
        cfg_path = "deploy/configs/inference_cls.yaml"

    __dir__ = os.path.dirname(__file__)
    cfg_path = os.path.join(__dir__, cfg_path)
    cfg = config.get_config(
        cfg_path, overrides=kwargs.get("override", None), show=False)
    if cfg.Global.get("inference_model_dir"):
        cfg.Global.inference_model_dir = inference_model_dir
    else:
        cfg.Global.rec_inference_model_dir = os.path.join(
            inference_model_dir,
            "PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0")
        cfg.Global.det_inference_model_dir = os.path.join(
            inference_model_dir, "picodet_PPLCNet_x2_5_mainbody_lite_v1.0")

    if "batch_size" in kwargs and kwargs["batch_size"]:
        cfg.Global.batch_size = kwargs["batch_size"]

    if "use_gpu" in kwargs and kwargs["use_gpu"] is not None:
        cfg.Global.use_gpu = kwargs["use_gpu"]
    if cfg.Global.use_gpu and not paddle.device.is_compiled_with_cuda():
        msg = "The current running environment does not support the use of GPU. CPU has been used instead."
        logger.warning(msg)
        cfg.Global.use_gpu = False

    if "infer_imgs" in kwargs and kwargs["infer_imgs"]:
        cfg.Global.infer_imgs = kwargs["infer_imgs"]
    if "index_dir" in kwargs and kwargs["index_dir"]:
        cfg.IndexProcess.index_dir = kwargs["index_dir"]
    if "data_file" in kwargs and kwargs["data_file"]:
        cfg.IndexProcess.data_file = kwargs["data_file"]
    if "enable_mkldnn" in kwargs and kwargs["enable_mkldnn"] is not None:
        cfg.Global.enable_mkldnn = kwargs["enable_mkldnn"]
    if "cpu_num_threads" in kwargs and kwargs["cpu_num_threads"]:
        cfg.Global.cpu_num_threads = kwargs["cpu_num_threads"]
    if "use_fp16" in kwargs and kwargs["use_fp16"] is not None:
        cfg.Global.use_fp16 = kwargs["use_fp16"]
    if "use_tensorrt" in kwargs and kwargs["use_tensorrt"] is not None:
        cfg.Global.use_tensorrt = kwargs["use_tensorrt"]
    if "gpu_mem" in kwargs and kwargs["gpu_mem"]:
        cfg.Global.gpu_mem = kwargs["gpu_mem"]
    if "resize_short" in kwargs and kwargs["resize_short"]:
        cfg.PreProcess.transform_ops[0]["ResizeImage"][
            "resize_short"] = kwargs["resize_short"]
    if "crop_size" in kwargs and kwargs["crop_size"]:
        cfg.PreProcess.transform_ops[1]["CropImage"]["size"] = kwargs[
            "crop_size"]

    # TODO(gaotingquan): not robust
    if cfg.get("PostProcess"):
        if "Topk" in cfg.PostProcess:
            if "topk" in kwargs and kwargs["topk"]:
                cfg.PostProcess.Topk.topk = kwargs["topk"]
            if "class_id_map_file" in kwargs and kwargs["class_id_map_file"]:
                cfg.PostProcess.Topk.class_id_map_file = kwargs[
                    "class_id_map_file"]
            else:
                class_id_map_file_path = os.path.relpath(
                    cfg.PostProcess.Topk.class_id_map_file, "../")
                cfg.PostProcess.Topk.class_id_map_file = os.path.join(
                    __dir__, class_id_map_file_path)
        if "ThreshOutput" in cfg.PostProcess:
            if "thresh" in kwargs and kwargs["thresh"]:
                cfg.PostProcess.ThreshOutput.thresh = kwargs["thresh"]
            if "class_id_map_file" in kwargs and kwargs["class_id_map_file"]:
                cfg.PostProcess.ThreshOutput["class_id_map_file"] = kwargs[
                    "class_id_map_file"]
            elif "class_id_map_file" in cfg.PostProcess.ThreshOutput:
                class_id_map_file_path = os.path.relpath(
                    cfg.PostProcess.ThreshOutput.class_id_map_file, "../")
                cfg.PostProcess.ThreshOutput.class_id_map_file = os.path.join(
                    __dir__, class_id_map_file_path)
        if "VehicleAttribute" in cfg.PostProcess:
            if "color_threshold" in kwargs and kwargs["color_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "color_threshold"]
            if "type_threshold" in kwargs and kwargs["type_threshold"]:
                cfg.PostProcess.VehicleAttribute.type_threshold = kwargs[
                    "type_threshold"]
        if "TableAttribute" in cfg.PostProcess:
            if "source_threshold" in kwargs and kwargs["source_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "source_threshold"]
            if "number_threshold" in kwargs and kwargs["number_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "number_threshold"]
            if "color_threshold" in kwargs and kwargs["color_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "color_threshold"]
            if "clarity_threshold" in kwargs and kwargs["clarity_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "clarity_threshold"]
            if "obstruction_threshold" in kwargs and kwargs[
                    "obstruction_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "obstruction_threshold"]
            if "angle_threshold" in kwargs and kwargs["angle_threshold"]:
                cfg.PostProcess.VehicleAttribute.color_threshold = kwargs[
                    "angle_threshold"]
    if "save_dir" in kwargs and kwargs["save_dir"]:
        cfg.PostProcess.SavePreLabel.save_dir = kwargs["save_dir"]

    return cfg


def args_cfg():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer_imgs",
        type=str,
        required=False,
        help="The image(s) to be predicted.")
    parser.add_argument(
        "--model_name", type=str, help="The model name to be used.")
    parser.add_argument(
        "--predict_type",
        type=str,
        default="cls",
        help="The predict type to be selected.")
    parser.add_argument(
        "--inference_model_dir",
        type=str,
        help="The directory of model files. Valid when model_name not specifed."
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=False,
        help="The index directory path.")
    parser.add_argument(
        "--data_file", type=str, required=False, help="The label file path.")
    parser.add_argument("--use_gpu", type=str2bool, help="Whether use GPU.")
    parser.add_argument(
        "--gpu_mem",
        type=int,
        help="The memory size of GPU allocated to predict.")
    parser.add_argument(
        "--enable_mkldnn",
        type=str2bool,
        help="Whether use MKLDNN. Valid when use_gpu is False")
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        help="The threads number when predicting on CPU.")
    parser.add_argument(
        "--use_tensorrt",
        type=str2bool,
        help="Whether use TensorRT to accelerate.")
    parser.add_argument(
        "--use_fp16", type=str2bool, help="Whether use FP16 to predict.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--topk",
        type=int,
        help="Return topk score(s) and corresponding results when Topk postprocess is used."
    )
    parser.add_argument(
        "--class_id_map_file",
        type=str,
        help="The path of file that map class_id and label.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="The threshold of ThreshOutput when postprocess is used.")
    parser.add_argument("--color_threshold", type=float, help="")
    parser.add_argument("--type_threshold", type=float, help="")
    parser.add_argument(
        "--save_dir",
        type=str,
        help="The directory to save prediction results as pre-label.")
    parser.add_argument(
        "--resize_short", type=int, help="Resize according to short size.")
    parser.add_argument("--crop_size", type=int, help="Centor crop size.")
    parser.add_argument(
        "--build_gallery",
        type=str2bool,
        default=False,
        help="Whether build gallery.")
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return vars(args)


def print_info():
    """Print list of supported models in formatted.
    """
    imn_table = PrettyTable(["IMN Model Series", "Model Name"])
    pulc_table = PrettyTable(["PULC Models"])
    shitu_table = PrettyTable(["PP-ShiTu Models"])
    try:
        sz = os.get_terminal_size()
        total_width = sz.columns
        first_width = 30
        second_width = total_width - first_width if total_width > 50 else 10
    except OSError:
        total_width = 100
        second_width = 100
    for series in IMN_MODEL_SERIES:
        names = textwrap.fill(
            "  ".join(IMN_MODEL_SERIES[series]), width=second_width)
        imn_table.add_row([series, names])

    table_width = len(str(imn_table).split("\n")[0])
    pulc_table.add_row([
        textwrap.fill(
            "  ".join(PULC_MODELS), width=total_width).center(table_width - 4)
    ])
    shitu_table.add_row([
        textwrap.fill(
            "  ".join(SHITU_MODELS), width=total_width).center(table_width - 4)
    ])

    print("{}".format("-" * table_width))
    print("Models supported by PaddleClas".center(table_width))
    print(imn_table)
    print(pulc_table)
    print(shitu_table)
    print("Powered by PaddlePaddle!".rjust(table_width))
    print("{}".format("-" * table_width))


def get_imn_model_names():
    """Get the model names list.
    """
    model_names = []
    for series in IMN_MODEL_SERIES:
        model_names += (IMN_MODEL_SERIES[series])
    return model_names


def similar_model_names(name="", names=[], thresh=0.1, topk=5):
    """Find the most similar topk model names.
    """
    scores = []
    for idx, n in enumerate(names):
        if n.startswith("__"):
            continue
        score = SequenceMatcher(None, n.lower(), name.lower()).quick_ratio()
        if score > thresh:
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    similar_names = [names[s[0]] for s in scores[:min(topk, len(scores))]]
    return similar_names


def download_with_progressbar(url, save_path):
    """Download from url with progressbar.
    """
    if os.path.isfile(save_path):
        os.remove(save_path)
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes or not os.path.isfile(
            save_path):
        raise Exception(
            f"Something went wrong while downloading file from {url}")


def check_model_file(model_type, model_name):
    """Check the model files exist and download and untar when no exist.
    """
    if model_type == "pulc":
        storage_directory = partial(os.path.join, BASE_INFERENCE_MODEL_DIR,
                                    "PULC", model_name)
        url = PULC_MODEL_BASE_DOWNLOAD_URL.format(model_name)
    elif model_type == "shitu":
        storage_directory = partial(os.path.join, BASE_INFERENCE_MODEL_DIR,
                                    "PP-ShiTu", model_name)
        url = SHITU_MODEL_BASE_DOWNLOAD_URL.format(model_name)
    else:
        storage_directory = partial(os.path.join, BASE_INFERENCE_MODEL_DIR,
                                    "IMN", model_name)
        url = IMN_MODEL_BASE_DOWNLOAD_URL.format(model_name)

    tar_file_name_list = [
        "inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel"
    ]
    model_file_path = storage_directory("inference.pdmodel")
    params_file_path = storage_directory("inference.pdiparams")
    if not os.path.exists(model_file_path) or not os.path.exists(
            params_file_path):
        tmp_path = storage_directory(url.split("/")[-1])
        logger.info(f"download {url} to {tmp_path}")
        os.makedirs(storage_directory(), exist_ok=True)
        download_with_progressbar(url, tmp_path)
        with tarfile.open(tmp_path, "r") as tarObj:
            for member in tarObj.getmembers():
                filename = None
                for tar_file_name in tar_file_name_list:
                    if tar_file_name in member.name:
                        filename = tar_file_name
                if filename is None:
                    continue
                file = tarObj.extractfile(member)
                with open(storage_directory(filename), "wb") as f:
                    f.write(file.read())
        os.remove(tmp_path)
    if not os.path.exists(model_file_path) or not os.path.exists(
            params_file_path):
        raise Exception(
            f"Something went wrong while praparing the model[{model_name}] files!"
        )

    return storage_directory()


class PaddleClas(object):
    """PaddleClas.
    """

    def __init__(self,
                 build_gallery: bool=False,
                 gallery_image_root: str=None,
                 gallery_data_file: str=None,
                 index_dir: str=None,
                 model_name: str=None,
                 inference_model_dir: str=None,
                 **kwargs):
        """Init PaddleClas with config.

        Args:
            model_name (str, optional): The model name supported by PaddleClas. If specified, override config. Defaults to None.
            inference_model_dir (str, optional): The directory that contained model file and params file to be used. If specified, override config. Defaults to None.
            use_gpu (bool, optional): Whether use GPU. If specified, override config. Defaults to True.
            batch_size (int, optional): The batch size to pridict. If specified, override config. Defaults to 1.
            topk (int, optional): Return the top k prediction results with the highest score. Defaults to 5.
        """
        super().__init__()

        if build_gallery:
            self.model_type, inference_model_dir = self._check_input_model(
                model_name
                if model_name else "PP-ShiTuV2", inference_model_dir)
            self._config = init_config(self.model_type, model_name
                                       if model_name else "PP-ShiTuV2",
                                       inference_model_dir, **kwargs)
            if gallery_image_root:
                self._config.IndexProcess.image_root = gallery_image_root
            if gallery_data_file:
                self._config.IndexProcess.data_file = gallery_data_file
            if index_dir:
                self._config.IndexProcess.index_dir = index_dir

            logger.info("Building Gallery...")
            GalleryBuilder(self._config)

        else:
            self.model_type, inference_model_dir = self._check_input_model(
                model_name, inference_model_dir)
            self._config = init_config(self.model_type, model_name,
                                       inference_model_dir, **kwargs)

            if self.model_type == "shitu":
                if index_dir:
                    self._config.IndexProcess.index_dir = index_dir
                self.predictor = SystemPredictor(self._config)
            else:
                self.predictor = ClsPredictor(self._config)

    def get_config(self):
        """Get the config.
        """
        return self._config

    def _check_input_model(self, model_name, inference_model_dir):
        """Check input model name or model files.
        """
        all_imn_model_names = get_imn_model_names()
        all_pulc_model_names = PULC_MODELS
        all_shitu_model_names = SHITU_MODELS

        if model_name:
            if model_name in all_imn_model_names:
                inference_model_dir = check_model_file("imn", model_name)
                return "imn", inference_model_dir
            elif model_name in all_pulc_model_names:
                inference_model_dir = check_model_file("pulc", model_name)
                return "pulc", inference_model_dir
            elif model_name in all_shitu_model_names:
                inference_model_dir = check_model_file(
                    "shitu",
                    "PP-ShiTuV2/general_PPLCNetV2_base_pretrained_v1.0")
                inference_model_dir = check_model_file(
                    "shitu", "picodet_PPLCNet_x2_5_mainbody_lite_v1.0")
                inference_model_dir = os.path.abspath(
                    os.path.dirname(inference_model_dir))
                return "shitu", inference_model_dir
            else:
                similar_imn_names = similar_model_names(model_name,
                                                        all_imn_model_names)
                similar_pulc_names = similar_model_names(model_name,
                                                         all_pulc_model_names)
                similar_names_str = ", ".join(similar_imn_names +
                                              similar_pulc_names)
                err = f"{model_name} is not provided by PaddleClas. \nMaybe you want the : [{similar_names_str}]. \nIf you want to use your own model, please specify inference_model_dir!"
                raise InputModelError(err)
        elif inference_model_dir:
            model_file_path = os.path.join(inference_model_dir,
                                           "inference.pdmodel")
            params_file_path = os.path.join(inference_model_dir,
                                            "inference.pdiparams")
            if not os.path.isfile(model_file_path) or not os.path.isfile(
                    params_file_path):
                err = f"There is no model file or params file in this directory: {inference_model_dir}"
                raise InputModelError(err)
            return "custom", inference_model_dir
        else:
            err = "Please specify the model name supported by PaddleClas or directory contained model files(inference.pdmodel, inference.pdiparams)."
            raise InputModelError(err)
        return None

    def predict_cls(self,
                    input_data: Union[str, np.array],
                    print_pred: bool=False) -> Generator[list, None, None]:
        """Predict input_data.

        Args:
            input_data (Union[str, np.array]):
                When the type is str, it is the path of image, or the directory containing images, or the URL of image from Internet.
                When the type is np.array, it is the image data whose channel order is RGB.
            print_pred (bool, optional): Whether print the prediction result. Defaults to False.

        Raises:
            ImageTypeError: Illegal input_data.

        Yields:
            Generator[list, None, None]:
                The prediction result(s) of input_data by batch_size. For every one image,
                prediction result(s) is zipped as a dict, that includs topk "class_ids", "scores" and "label_names".
                The format of batch prediction result(s) is as follow: [{"class_ids": [...], "scores": [...], "label_names": [...]}, ...]
        """

        if isinstance(input_data, np.ndarray):
            yield self.predictor.predict(input_data)
        elif isinstance(input_data, str):
            if input_data.startswith("http") or input_data.startswith("https"):
                image_storage_dir = partial(os.path.join, BASE_IMAGES_DIR)
                if not os.path.exists(image_storage_dir()):
                    os.makedirs(image_storage_dir())
                image_save_path = image_storage_dir("tmp.jpg")
                download_with_progressbar(input_data, image_save_path)
                logger.info(
                    f"Image to be predicted from Internet: {input_data}, has been saved to: {image_save_path}"
                )
                input_data = image_save_path
            image_list = get_image_list(input_data)

            batch_size = self._config.Global.get("batch_size", 1)

            img_list = []
            img_path_list = []
            cnt = 0
            for idx_img, img_path in enumerate(image_list):
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(
                        f"Image file failed to read and has been skipped. The path: {img_path}"
                    )
                    continue
                img = img[:, :, ::-1]
                img_list.append(img)
                img_path_list.append(img_path)
                cnt += 1

                if cnt % batch_size == 0 or (idx_img + 1) == len(image_list):
                    preds = self.predictor.predict(img_list)

                    if preds:
                        for idx_pred, pred in enumerate(preds):
                            pred["filename"] = img_path_list[idx_pred]
                            if print_pred:
                                logger.info(", ".join(
                                    [f"{k}: {pred[k]}" for k in pred]))

                    img_list = []
                    img_path_list = []
                    yield preds
        else:
            err = "Please input legal image! The type of image supported by PaddleClas are: NumPy.ndarray and string of local path or Ineternet URL"
            raise ImageTypeError(err)
        return

    def predict_shitu(self,
                      input_data: Union[str, np.array],
                      print_pred: bool=False) -> Generator[list, None, None]:
        """Predict input_data.
        Args:
            input_data (Union[str, np.array]):
                When the type is str, it is the path of image, or the directory containing images, or the URL of image from Internet.
                When the type is np.array, it is the image data whose channel order is RGB.
            print_pred (bool, optional): Whether print the prediction result. Defaults to False.

        Raises:
            ImageTypeError: Illegal input_data.

        Yields:
            Generator[list, None, None]:
                The prediction result(s) of input_data by batch_size. For every one image,
                prediction result(s) is zipped as a dict, that includs topk "class_ids", "scores" and "label_names".
                The format of batch prediction result(s) is as follow: [{"class_ids": [...], "scores": [...], "label_names": [...]}, ...]
        """
        if input_data is None and self._config.Global.infer_imgs:
            input_data = self._config.Global.infer_imgs

        if isinstance(input_data, np.ndarray):
            yield self.predictor.predict(input_data)
        elif isinstance(input_data, str):
            if input_data.startswith("http") or input_data.startswith("https"):
                image_storage_dir = partial(os.path.join, BASE_IMAGES_DIR)
                if not os.path.exists(image_storage_dir()):
                    os.makedirs(image_storage_dir())
                image_save_path = image_storage_dir("tmp.jpg")
                download_with_progressbar(input_data, image_save_path)
                logger.info(
                    f"Image to be predicted from Internet: {input_data}, has been saved to: {image_save_path}"
                )
                input_data = image_save_path
            image_list = get_image_list(input_data)

            cnt = 0
            for idx_img, img_path in enumerate(image_list):
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(
                        f"Image file failed to read and has been skipped. The path: {img_path}"
                    )
                    continue
                img = img[:, :, ::-1]
                cnt += 1

                preds = self.predictor.predict(
                    img)  # [dict1, dict2, ..., dictn]
                if preds:
                    if print_pred:
                        logger.info(f"{preds}, filename: {img_path}")

                yield preds
        else:
            err = "Please input legal image! The type of image supported by PaddleClas are: NumPy.ndarray and string of local path or Ineternet URL"
            raise ImageTypeError(err)
        return

    def predict(self,
                input_data: Union[str, np.array],
                print_pred: bool=False,
                predict_type="cls"):
        assert predict_type in ["cls", "shitu"
                                ], "Predict type should be 'cls' or 'shitu'."
        if predict_type == "cls":
            return self.predict_cls(input_data, print_pred)
        elif predict_type == "shitu":
            assert not isinstance(input_data, (
                list, tuple
            )), "PP-ShiTu predictor only support single image as input now."
            return self.predict_shitu(input_data, print_pred)
        else:
            raise ModuleNotFoundError


# for CLI
def main():
    """Function API used for commad line.
    """
    print_info()
    cfg = args_cfg()
    clas_engine = PaddleClas(**cfg)
    if cfg["build_gallery"] == False:
        res = clas_engine.predict(
            cfg["infer_imgs"],
            print_pred=True,
            predict_type=cfg["predict_type"])
        for _ in res:
            pass
        logger.info("Predict complete!")
    return


if __name__ == "__main__":
    main()
