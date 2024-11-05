# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial
import io
import six
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION
from paddle.vision.transforms import ColorJitter as RawColorJitter
from paddle.vision.transforms import CenterCrop, Resize
from paddle.vision.transforms import RandomRotation as RawRandomRotation
from paddle.vision.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
from paddle.vision.transforms import functional as F
from .autoaugment import ImageNetPolicy
from .functional import augmentations
from ppcls.utils import logger


def format_data(func):
    def warpper(self, data):
        if isinstance(data, dict):
            img = data["img"]
            result = func(self, img)
            if not isinstance(result, dict):
                result = {"img": result}
            return { ** data, ** result}
        else:
            result = func(self, data)
            if isinstance(result, dict):
                result = result["img"]
            return result

    return warpper


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2", return_numpy=True):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'random': (cv2.INTER_LINEAR, cv2.INTER_CUBIC)
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING,
            'random': (Image.BILINEAR, Image.BICUBIC)
        }

        def _cv2_resize(src, size, resample):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            return cv2.resize(src, size, interpolation=resample)

        def _pil_resize(src, size, resample, return_numpy=True):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            if isinstance(src, np.ndarray):
                pil_img = Image.fromarray(src)
            else:
                pil_img = src
            pil_img = pil_img.resize(size, resample)
            if return_numpy:
                return np.asarray(pil_img)
            return pil_img

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(_cv2_resize, resample=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            elif interpolation is None:
                interpolation = Image.BILINEAR
            self.resize_func = partial(
                _pil_resize, resample=interpolation, return_numpy=return_numpy)
        else:
            logger.warning(
                f"The backend of Resize only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. Use \"cv2\" instead."
            )
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        if isinstance(size, list):
            size = tuple(size)
        return self.resize_func(src, size)


class RandomInterpolationAugment(object):
    def __init__(self, prob):
        self.prob = prob

    def _aug(self, img):
        img_shape = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice([
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        ])
        small_img = cv2.resize(
            img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice([
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        ])
        aug_img = cv2.resize(
            small_img, (img_shape[1], img_shape[0]),
            interpolation=interpolation)
        return aug_img

    def __call__(self, img):
        if np.random.random() < self.prob:
            if isinstance(img, np.ndarray):
                return self._aug(img)
            else:
                pil_img = np.array(img)
                aug_img = self._aug(pil_img)
                img = Image.fromarray(aug_img.astype(np.uint8))
                return img
        else:
            return img


class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass


class DecodeImage(object):
    """ decode image """

    def __init__(self,
                 to_np=True,
                 to_rgb=True,
                 channel_first=False,
                 backend="cv2"):
        self.to_np = to_np  # to numpy
        self.to_rgb = to_rgb  # only enabled when to_np is True
        self.channel_first = channel_first  # only enabled when to_np is True

        if backend.lower() not in ["cv2", "pil"]:
            logger.warning(
                f"The backend of DecodeImage only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. Use \"cv2\" instead."
            )
            backend = "cv2"
        self.backend = backend.lower()

        if not to_np:
            logger.warning(
                f"\"to_rgb\" and \"channel_first\" are only enabled when to_np is True. \"to_np\" is now {to_np}."
            )

    @format_data
    def __call__(self, img):
        if isinstance(img, Image.Image):
            assert self.backend == "pil", "invalid input 'img' in DecodeImage"
        elif isinstance(img, np.ndarray):
            assert self.backend == "cv2", "invalid input 'img' in DecodeImage"
        elif isinstance(img, bytes):
            if self.backend == "pil":
                data = io.BytesIO(img)
                img = Image.open(data).convert("RGB")
            else:
                data = np.frombuffer(img, dtype="uint8")
                img = cv2.imdecode(data, 1)
        else:
            raise ValueError("invalid input 'img' in DecodeImage")

        if self.to_np:
            if self.backend == "pil":
                assert img.mode == "RGB", f"invalid mode of image[{img.mode}]"
                img = np.asarray(img)[:, :, ::-1]  # BRG

            if self.to_rgb:
                assert img.shape[
                    2] == 3, f"invalid shape of image[{img.shape}]"
                img = img[:, :, ::-1]

            if self.channel_first:
                img = img.transpose((2, 0, 1))
        return img


class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2",
                 return_numpy=True):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation,
            backend=backend,
            return_numpy=return_numpy)

    @format_data
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = img.size

        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class CropWithPadding(RandomResizedCrop):
    """
    crop image and padding to original size
    """

    def __init__(self,
                 prob=1,
                 padding_num=0,
                 size=224,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation='bilinear',
                 key=None):
        super().__init__(size, scale, ratio, interpolation, key)
        self.prob = prob
        self.padding_num = padding_num

    def __call__(self, img):
        if np.random.random() < self.prob:
            # RandomResizedCrop augmentation
            new = np.zeros_like(np.array(img)) + self.padding_num
            #  orig_W, orig_H = F._get_image_size(sample)
            i, j, h, w = self._dynamic_get_param(img)
            cropped = F.crop(img, i, j, h, w)
            new[i:i + h, j:j + w, :] = np.array(cropped)
            return new
        else:
            return img

    def _get_image_size(self, img):
        if F._is_pil_image(img):
            return img.size
        elif F._is_numpy_image(img):
            return img.shape[:2][::-1]
        elif F._is_tensor_image(img):
            return img.shape[1:][::-1]  # chw
        else:
            raise TypeError("Unexpected type {}".format(type(img)))


class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class CropImageAtRatio(object):
    """ crop image with specified size and padding"""

    def __init__(self, size: int, pad: int, interpolation="bilinear"):
        self.size = size
        self.ratio = size / (size + pad)
        self.interpolation = interpolation

    def __call__(self, img):
        height, width = img.shape[:2]
        crop_size = int(self.ratio * min(height, width))

        y = (height - crop_size) // 2
        x = (width - crop_size) // 2

        crop_img = img[y:y + crop_size, x:x + crop_size, :]
        return F.resize(crop_img, [self.size, self.size], self.interpolation)


class Padv2(object):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, list): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """

        if not isinstance(size, (int, list)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def __call__(self, img):
        im_h, im_w = img.shape[:2]
        if self.size:
            w, h = self.size
            assert (
                im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            return img.astype(np.float32)

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        return self.apply_image(img, offsets, im_size, size)


class RandomCropImage(object):
    """Random crop image only
    """

    def __init__(self, size):
        super(RandomCropImage, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size

    def __call__(self, img):

        h, w = img.shape[:2]
        tw, th = self.size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        img = img[i:i + th, j:j + tw, :]
        return img


class RandCropImage(object):
    """ random crop image """

    def __init__(self,
                 size,
                 progress_size=None,
                 scale=None,
                 ratio=None,
                 interpolation=None,
                 use_log_aspect=False,
                 backend="cv2"):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.progress_size = progress_size
        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio
        self.use_log_aspect = use_log_aspect

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    @format_data
    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        if self.use_log_aspect:
            log_ratio = list(map(math.log, ratio))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        img_h, img_w = img.shape[:2]
        bound = min((float(img_w) / img_h) / aspect_ratio,
                    (float(img_h) / img_w) * aspect_ratio)
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        w = int(math.sqrt(target_area * aspect_ratio))
        h = int(math.sqrt(target_area / aspect_ratio))

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = self._resize_func(img[j:j + h, i:i + w, :], size)
        return img


class RandCropImageV2(object):
    """ RandCropImageV2 is different from RandCropImage,
    it will Select a cutting position randomly in a uniform distribution way,
    and cut according to the given size without resize at last."""

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[0], img.shape[1]
        else:
            img_w, img_h = img.size
        tw, th = self.size

        if img_h + 1 < th or img_w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".
                format((th, tw), (img_h, img_w)))

        if img_w == tw and img_h == th:
            return img

        top = random.randint(0, img_h - th + 1)
        left = random.randint(0, img_w - tw + 1)
        if isinstance(img, np.ndarray):
            return img[top:top + th, left:left + tw, :]
        else:
            return img.crop((left, top, left + tw, top + th))


class RandFlipImage(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    @format_data
    def __call__(self, img):
        if random.randint(0, 1) == 1:
            if isinstance(img, np.ndarray):
                return cv2.flip(img, self.flip_code)
            else:
                if self.flip_code == 1:
                    return img.transpose(Image.FLIP_LEFT_RIGHT)
                elif self.flip_code == 0:
                    return img.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(
                        Image.FLIP_LEFT_RIGHT)
        else:
            return img


class AutoAugment(object):
    def __init__(self):
        self.policy = ImageNetPolicy()

    def __call__(self, img):
        from PIL import Image
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        img = self.policy(img)
        img = np.asarray(img)


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    @format_data
    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))

        img = img.astype(self.output_dtype)
        return img


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self,
                 prob=0.5,
                 aug_prob_coeff=0.1,
                 mixture_width=3,
                 mixture_depth=1,
                 aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.augmentations = augmentations
        # fmt: on

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.
        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(
            np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        # image = Image.fromarray(image)
        mix = np.zeros(image.shape)
        for i in range(self.mixture_width):
            image_aug = image.copy()
            image_aug = Image.fromarray(image_aug)
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)


class ColorJitter(RawColorJitter):
    """ColorJitter.
    """

    def __init__(self, prob=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            if not isinstance(img, Image.Image):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
            img = super()._apply_image(img)
            if isinstance(img, Image.Image):
                img = np.asarray(img)
        return img


class RandomRotation(RawRandomRotation):
    """RandomRotation.
    """

    def __init__(self, prob=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            img = super()._apply_image(img)
        return img


class Pad(object):
    """
    Pads the given PIL.Image on all sides with specified padding mode and fill value.
    adapted from: https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Pad
    """

    def __init__(self,
                 padding: int,
                 fill: int=0,
                 padding_mode: str="constant",
                 backend: str="pil"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.backend = backend
        assert backend in [
            "pil", "cv2"
        ], f"backend must in ['pil', 'cv2'], but got {backend}"

    def _parse_fill(self, fill, img, min_pil_version, name="fillcolor"):
        # Process fill color for affine transforms
        major_found, minor_found = (int(v)
                                    for v in PILLOW_VERSION.split('.')[:2])
        major_required, minor_required = (
            int(v) for v in min_pil_version.split('.')[:2])
        if major_found < major_required or (major_found == major_required and
                                            minor_found < minor_required):
            if fill is None:
                return {}
            else:
                msg = (
                    "The option to fill background area of the transformed image, "
                    "requires pillow>={}")
                raise RuntimeError(msg.format(min_pil_version))

        num_bands = len(img.getbands())
        if fill is None:
            fill = 0
        if isinstance(fill, (int, float)) and num_bands > 1:
            fill = tuple([fill] * num_bands)
        if isinstance(fill, (list, tuple)):
            if len(fill) != num_bands:
                msg = (
                    "The number of elements in 'fill' does not match the number of "
                    "bands of the image ({} != {})")
                raise ValueError(msg.format(len(fill), num_bands))

            fill = tuple(fill)

        return {name: fill}

    def __call__(self, img):
        if self.backend == "pil":
            opts = self._parse_fill(self.fill, img, "2.3.0", name="fill")
            if img.mode == "P":
                palette = img.getpalette()
                img = ImageOps.expand(img, border=self.padding, **opts)
                img.putpalette(palette)
                return img
            return ImageOps.expand(img, border=self.padding, **opts)
        else:
            img = cv2.copyMakeBorder(
                img,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                cv2.BORDER_CONSTANT,
                value=(self.fill, self.fill, self.fill))
            return img


# TODO(gaotingquan): integrate into RandomRotation
class RandomRot90(object):
    """RandomRot90
    """

    def __init__(self):
        pass

    @format_data
    def __call__(self, img):
        orientation = random.choice([0, 1, 2, 3])
        if orientation:
            img = np.rot90(img, orientation)
        return {"img": img, "random_rot90_orientation": orientation}


class BlurImage(object):
    """BlurImage
    """

    def __init__(self,
                 ratio=0.5,
                 motion_max_ksize=12,
                 motion_max_angle=45,
                 gaussian_max_ksize=12):
        self.ratio = ratio
        self.motion_max_ksize = motion_max_ksize
        self.motion_max_angle = motion_max_angle
        self.gaussian_max_ksize = gaussian_max_ksize

    def _gaussian_blur(self, img, max_ksize=12):
        ksize = (np.random.choice(np.arange(5, max_ksize, 2)),
                 np.random.choice(np.arange(5, max_ksize, 2)))
        img = cv2.GaussianBlur(img, ksize, 0)
        return img

    def _motion_blur(self, img, max_ksize=12, max_angle=45):
        degree = np.random.choice(np.arange(5, max_ksize, 2))
        angle = np.random.choice(np.arange(-1 * max_angle, max_angle))

        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M,
                                            (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)

        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        img = np.array(blurred, dtype=np.uint8)
        return img

    @format_data
    def __call__(self, img):
        if random.random() > self.ratio:
            label = 0
        else:
            method = random.choice(["gaussian", "motion"])
            if method == "gaussian":
                img = self._gaussian_blur(img, self.gaussian_max_ksize)
            else:
                img = self._motion_blur(img, self.motion_max_ksize,
                                        self.motion_max_angle)
            label = 1
        return {"img": img, "blur_image": label}


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PCALighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    reference: https://github.com/DingXiaoH/DiverseBranchBlock
    """

    def __init__(self):
        self.alphastd = 0.1
        self.eigval = [0.2175, 0.0188, 0.0045]
        self.eigvec = [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
        self.eigval = np.array(self.eigval).astype(np.float32)
        self.eigvec = np.array(self.eigvec).astype(np.float32)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        img = img.transpose((2, 0, 1))
        alpha = np.random.normal(0, self.alphastd, size=(3)).astype(np.float32)
        rgb = self.eigvec * np.broadcast_to(alpha.reshape(1, 3), (
            3, 3)) * np.broadcast_to(self.eigval.reshape(1, 3), (3, 3))
        rgb = rgb.sum(1).squeeze()
        img = img + rgb.reshape(3, 1, 1)
        return img.transpose((1, 2, 0))
