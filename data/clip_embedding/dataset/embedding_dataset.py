# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-05-24 17:49:25


from functools import partial
import cv2
import PIL
import albumentations
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torchvision.transforms.functional as TF
from PIL import Image
import json

from clip_embedding.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

from .tsv_io import TSVFile
from .bin_file import BinFile
from .io_common import img_from_base64

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
    ])


class EmbeddingDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''

    def __init__(self, tsv_file, bin_file, map_file, **kwargs):
        """
        Args:
            tsv_file (str): path to tsv file
            bin_file (str): path to bin file
            map_file (str): path to map file
            kwargs: if 'tokenizer' is in kwargs, use it to tokenize text
                    if 'text_encoder' is in kwargs, use text-encoder (api, provided by server) to tokenize and encode text. If 'text_encoder' is provided, 'tokenizer' is ignored.
        """
        self.tsv = TSVFile(tsv_file)
        self.bin = BinFile(bin_file, map_file)

        self.transforms = _transform(
            224) if 'transforms' not in kwargs else kwargs['transforms']
        self.tokenizer = kwargs.get('tokenizer', None)
        self.text_encoder = kwargs.get('text_encoder', None)
        assert not (
            self.tokenizer is not None and self.text_encoder is not None), "tokenizer and text_encoder cannot be both provided"

    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_label(self, row):
        label = json.loads(row[1])
        if 'text' in label:
            return label['text']
        elif 'caption' in label:
            return label['caption']
        else:
            raise ValueError("No text in label")

    def __getitem__(self, index):
        """
        Note:
            if self.text_encoder exists, return ([clip-]image embedding, [clip-]text embedding, img, text, text-encodigs)
            elif self.tokenizer exists, return ([clip-]image embedding, [clip-]text embedding, img, text, token, mask)
            else return ([clip-]image embedding, [clip-]text embedding, img, text)
        """
        embedding, idx = self.bin.seek(index)

        split = embedding.size // 2
        image_embedding = torch.tensor(embedding[:split])
        text_embedding = torch.tensor(embedding[split:])

        row = self.tsv.seek(idx)
        img = self.read_img(row)
        img = self.transforms(img)
        text = self.read_label(row)

        if self.text_encoder is not None:
            # TODO: adpat to text_encoder's encode api
            text_encodings = self.text_encoder.encode(text)
            return image_embedding, text_embedding, img, text, text_encodings
        if self.tokenizer is not None:
            token, mask = self.tokenizer(text)
            token, mask = torch.tensor(token), torch.tensor(mask)
            return image_embedding, text_embedding, img, text, token, mask
        else:
            return image_embedding, text_embedding, img, text

    def __len__(self):
        return self.bin.num_rows()


class SRDataset(Dataset):
    '''
        TSV dataset for tsv file

        Superresolution Dataset
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn
    '''

    def __init__(self, config, tsv_file, bin_file, map_file):
        """
        Args:
            config:
                image_size: image size read from tsv file
                downscale_factor: Low Resolution Downscale Factor
                min_crop_factor: determines crop size s, where s = c * min_img_side_len with c 
                                sampled from interval (min_crop_f, max_crop_f)
                max_crop_factor: see above
                random_crop: bool, enable random crop if True, else center crop
                degradation: str, ["bsrgan", "bsrgan_light", interpolation_mode etc]
            tsv_file (str): path to tsv file
            bin_file (str): path to bin file
            map_file (str): path to map file
        """
        self.tsv = TSVFile(tsv_file)
        self.bin = BinFile(bin_file, map_file)
        self.config = config
        self._init_bsrgan_degrade(config)

    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_label(self, row):
        label = json.loads(row[1])
        return label['text']
    
    def _init_bsrgan_degrade(self, config):
        self.size = config.image_size
        self.downscale_f = config.downscale_factor
        self.LR_size = self.size // self.downscale_f
        self.min_crop_f = config.min_crop_factor
        self.max_crop_f = config.max_crop_factor
        assert 0.0 <= self.min_crop_f <= self.max_crop_f <= 1.0, \
            "min_crop_f and max_crop_f must be between 0 and 1"
        self.center_crop = not config.random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=self.size, interpolation=cv2.INTER_AREA)

        # gets reset later if incase interp_op is from pillow
        self.pil_interpolation = False

        if config.degradation == "bsrgan":
            self.degradation_process = partial(
                degradation_fn_bsr, sf=self.downscale_f)
        elif config.degradation == "bsrgan_light":
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=self.downscale_f)
        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[config.degradation]
            self.pil_interpolation = config.degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize, size=[self.LR_size,self.LR_size], interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)

    def __getitem__(self, index):
        *_, idx = self.bin.seek(index)
        row = self.tsv.seek(idx)

        # read from tsv line
        img = self.read_img(row)
        img = np.array(img).astype(np.uint8)

        # crop
        min_side_len = min(img.shape[:2])
        crop_side_len = min_side_len * \
            np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = max(int(crop_side_len), 1)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(
                height=crop_side_len, width=crop_side_len)
        else:
            self.cropper = albumentations.RandomCrop(
                height=crop_side_len, width=crop_side_len)
        
        if self.config.flip is not None:
            img = albumentations.Flip(p=self.config.flip)(image=img)["image"]

        img = self.cropper(image=img)['image']
        img = self.image_rescaler(image=img)['image']

        # downsample interpolation
        if self.pil_interpolation:
            img_pil = PIL.Image.fromarray(img)
            LR_img = self.degradation_process(img_pil)
            LR_img = np.array(LR_img).astype(np.uint8)
        else:
            LR_img = self.degradation_process(image=img)['image']
        
        # apply gaussian blur noise
        if self.config.gaussian_blur:
            LR_img = albumentations.GaussianBlur(
                blur_limit=(3,3), sigma_limit=(0.6,0.6)
            )(image=LR_img)['image']

        img, LR_img = map(TF.to_tensor, [img, LR_img])
        return img, LR_img

    def __len__(self):
        return self.tsv.num_rows()
