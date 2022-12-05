# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-05-29 16:34:40

from functools import partial
import os
import PIL
import albumentations
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import torchvision.transforms.functional as TF
from PIL import Image
import json
import glob

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


class EmbeddingFolderDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''

    def __init__(self, tsv_folder, bin_folder, partition=-1, **kwargs):
        """
        Args:
            tsv_folder (str): path to tsv folder
            bin_folder (str): path to bin folder
            kwargs: 1. if 'tokenizer' is in kwargs, use it to tokenize text
                    2. if 'text_encoder' is in kwargs, use text-encoder (api, provided by server) to tokenize and encode text. If 'text_encoder' is provided, 'tokenizer' is ignored.
        """
        self.tsv_list = [TSVFile(file) for file in self.get_folder_file(
            os.path.join(tsv_folder, "*.tsv"), partition)]
        bin_map_file = self.get_folder_file(
            os.path.join(bin_folder, "*.*"), partition)
        assert len(bin_map_file) % 2 == 0
        self.bin = [BinFile(bin_map_file[i], bin_map_file[i+1])
                    for i in range(0, len(bin_map_file), 2)][0]

        assert sum([tsv.num_rows()
                   for tsv in self.tsv_list]) > self.bin.num_rows()

        self.num_rows = []
        s = 0
        for tsv in self.tsv_list:
            s += tsv.num_rows()
            self.num_rows.append(s)

        self.transforms = _transform(
            224) if 'transforms' not in kwargs else kwargs['transforms']
        self.tokenizer = kwargs.get('tokenizer', None)
        self.text_encoder = kwargs.get('text_encoder', None)
        assert not (
            self.tokenizer is not None and self.text_encoder is not None), "tokenizer and text_encoder cannot be both provided"

    def get_folder_file(self, folder_file, partition=-1):
        tsv_file_list = glob.glob(folder_file)
        tsv_file_list.sort()

        if partition >= 0:
            tsv_len = len(tsv_file_list)
            split_size = tsv_len // 32
            tsv_file_list = [tsv_file_list[i:i+split_size]
                             for i in range(0, tsv_len, split_size)][partition]

        return tsv_file_list

    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_label(self, row):
        label = json.loads(row[1])
        return label['objects'][0]['caption']

    def find_index(self, index):
        subset_index = np.searchsorted(self.num_rows, index, side='right')
        subset_data_index = (
            index - self.num_rows[subset_index-1]) if (subset_index > 0) else index
        return subset_index, subset_data_index

    def __getitem__(self, index):
        """
        Note:
            if self.text_encoder exists, return ([clip-]image embedding, [clip-]text embedding, img, text-encodigs)
            elif self.tokenizer exists, return ([clip-]image embedding, [clip-]text embedding, img, text, token, mask)
            else return ([clip-]image embedding, [clip-]text embedding, img, text)
        """
        embedding, idx = self.bin.seek(index)

        split = embedding.size // 2
        image_embedding = torch.tensor(embedding[:split])
        text_embedding = torch.tensor(embedding[split:])

        subset_index, subset_data_index = self.find_index(idx)

        row = self.tsv_list[subset_index].seek(subset_data_index)
        img = self.read_img(row)
        img = self.transforms(img)
        text = self.read_label(row)
        
        if self.text_encoder is not None:
            # TODO: adpat to text_encoder's encode api
            text_encodings = self.text_encoder.encode(text)
            return image_embedding, text_embedding, img, text_encodings
        if self.tokenizer is not None:
            token, mask = self.tokenizer(text)
            token, mask = torch.tensor(token), torch.tensor(mask)
            return image_embedding, text_embedding, img, text, token, mask
        else:
            return image_embedding, text_embedding, img, text

    def __len__(self):
        return self.bin.num_rows()


class SRFolderDataset(Dataset):
    '''
        TSV dataset for tsv file

        Superresolution Dataset
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn
    '''

    def __init__(self, config, tsv_folder, bin_folder, partition=-1, **kwargs):
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
        """
        # glob tsv/bin files
        self.tsv_list = [TSVFile(file) for file in self.get_folder_file(
            os.path.join(tsv_folder, "*.tsv"), partition)]
        bin_map_file = self.get_folder_file(
            os.path.join(bin_folder, "*.*"), partition)
        assert len(bin_map_file) % 2 == 0
        self.bin = [BinFile(bin_map_file[i], bin_map_file[i+1])
                    for i in range(0, len(bin_map_file), 2)][0]

        assert sum([tsv.num_rows()
                   for tsv in self.tsv_list]) > self.bin.num_rows()

        self.num_rows = []
        s = 0
        for tsv in self.tsv_list:
            s += tsv.num_rows()
            self.num_rows.append(s)
        
        # initialize image transforms
        self.config = config
        self._init_bsrgan_degrade(config)

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
                    TF.resize, size=self.LR_size, interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)
    
    def get_folder_file(self, folder_file, partition=-1):
        tsv_file_list = glob.glob(folder_file)
        tsv_file_list.sort()

        if partition >= 0:
            tsv_len = len(tsv_file_list)
            split_size = tsv_len // 32
            tsv_file_list = [tsv_file_list[i:i+split_size]
                             for i in range(0, tsv_len, split_size)][partition]

        return tsv_file_list

    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_label(self, row):
        label = json.loads(row[1])
        return label['objects'][0]['caption']
    
    def find_index(self, index):
        subset_index = np.searchsorted(self.num_rows, index, side='right')
        subset_data_index = (
            index - self.num_rows[subset_index-1]) if (subset_index > 0) else index
        return subset_index, subset_data_index

    def __getitem__(self, index):
        *_, idx = self.bin.seek(index)
        subset_index, subset_data_index = self.find_index(idx)

        # seek and fetch row
        row = self.tsv_list[subset_index].seek(subset_data_index)
        
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
        
        img = self.cropper(image=img)['image']
        img = self.image_rescaler(image=img)['image']
        
        # downsample interpolation
        if self.pil_interpolation:
            img_pil = PIL.Image.fromarray(img)
            LR_img = self.degradation_process(img_pil)
            LR_img = np.array(LR_img).astype(np.uint8)
        else:
            LR_img = self.degradation_process(image=img)['image']
        
        img, LR_img = map(TF.to_tensor, [img, LR_img])
        return img, LR_img

    def __len__(self):
        return self.bin.num_rows()