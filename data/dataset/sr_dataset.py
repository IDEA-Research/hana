# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
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
import json
import PIL
from PIL import Image
import cv2
import numpy as np
import albumentations
import random
from functools import partial

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from data.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

class SRImageProcessor(object):
    '''
        SuperResolution Image processing.

        Superresolution Dataset
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn
    '''

    def __init__(
        self, 
        image_size,
        downscale_factor,
        min_crop_factor,
        max_crop_factor,
        random_crop,
        degradation,
        gaussian_blur,
        blur_prob=0.5, 
        random_flip_prob=0.5,
        HR_transforms=None, 
        LR_transforms=None, 
    ):
        """
        Args:
            image_size: image size 
            downscale_factor: Low Resolution Downscale Factor
            min_crop_factor: determines crop size s, where s = c * min_img_side_len with c 
                            sampled from interval (min_crop_f, max_crop_f)
            max_crop_factor: see above
            random_crop: bool, enable random crop if True, else center crop
            degradation: str, ["bsrgan", "bsrgan_light", interpolation_mode etc]
            gaussian_blur: bool, enable gaussian blur if True
            HR_transforms: transforms to apply to High Resolution image
            LR_transforms: transforms to apply to Low Resolution image
            blur_prob: probability of applying gaussian blur
            random_flip_prob: probability of applying random horizontal flip
        """
        self.HR_transforms = HR_transforms
        self.LR_transforms = LR_transforms
        self.gaussian_blur = gaussian_blur
        self.blur_prob = blur_prob
        self.flip_prob = random_flip_prob

        if self.gaussian_blur:
            self.blur = T.transforms.GaussianBlur(3, 0.6)
        self.flip = T.RandomHorizontalFlip(p=1.)

        self.use_album = self.HR_transforms is None or self.LR_transforms is None
        if self.use_album:
            self._init_bsrgan_degrade(
                image_size, downscale_factor, min_crop_factor, max_crop_factor, random_crop, degradation
            )


    def _init_bsrgan_degrade(
        self, 
        image_size,
        downscale_factor,
        min_crop_factor,
        max_crop_factor,
        random_crop:bool,
        degradation:str,
    ):
        self.size = image_size
        self.downscale_f = downscale_factor
        self.LR_size = self.size // self.downscale_f
        self.min_crop_f = min_crop_factor
        self.max_crop_f = max_crop_factor
        assert 0.0 <= self.min_crop_f <= self.max_crop_f <= 1.0, \
            "min_crop_f and max_crop_f must be between 0 and 1"
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_AREA)

        # gets reset later if incase interp_op is from pillow
        self.pil_interpolation = False

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=self.downscale_f)
        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=self.downscale_f)
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
            }[degradation]
            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize, size=[self.LR_size,self.LR_size], interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)

    def __call__(self, img):
        if not self.use_album:
            HR_img = self.HR_transforms(img)
            LR_img = self.LR_transforms(img)
            # random flip
            if random.random() < self.flip_prob:
                HR_img = self.flip(HR_img)
                LR_img = self.flip(LR_img)
            # apply gaussian blur noise
            if self.gaussian_blur and random.random() < self.blur_prob:
                LR_img = self.blur(LR_img)
            return HR_img, LR_img
        else:
            img = np.array(img).astype(np.uint8)

            # crop
            min_side_len = min(img.shape[:2])
            crop_side_len = min_side_len * \
                np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = max(int(crop_side_len), 1)

            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
            else:
                self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
            
            if self.flip_prob is not None:
                img = albumentations.Flip(p=self.flip_prob)(image=img)["image"]

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
            if self.gaussian_blur:
                LR_img = albumentations.GaussianBlur(
                    blur_limit=(3,3), sigma_limit=(0.6,0.6)
                )(image=LR_img)['image']

            HR_img, LR_img = map(TF.to_tensor, [img, LR_img])
            return HR_img, LR_img

class SRDataset(Dataset):
    def __init__(
        self, 
        mapping_file,
        sr_image_processor:SRImageProcessor,
        return_clip_embedding=False, 
        return_t5_embedding=False,
    ) -> None:
        super(SRDataset, self).__init__()

        self.mapping = self._load_mapping(mapping_file)
        self.keys = list(self.mapping.keys())
        self.data_dir = os.path.dirname(mapping_file)
        self.sr_image_processor = sr_image_processor
        self.return_clip_embedding = return_clip_embedding
        self.return_t5_embedding = return_t5_embedding
    
    def _load_mapping(self, mapping_file):
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        return mapping

    def __len__(self):
        return len(self.keys)
    
    def read_bytes_embedding(self, embedding_path, dtype):
        with open(embedding_path, 'rb') as f:
            embedding = np.frombuffer(f.read(), dtype=dtype)
        return embedding
    
    def __getitem__(self, index):
        key = self.keys[index]
        img_path = self.mapping[key]['img_path']
        img_path = os.path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        HR_img, LR_img = self.sr_image_processor(img)
        caption = self.mapping[key]['caption']
        
        out = {}
        out['HR_img'] = HR_img
        out['LR_img'] = LR_img
        out['caption'] = caption
        
        if self.return_clip_embedding:
            text_embedding_path = os.path.join(
                self.data_dir, 'clip_embedding', f'{key}.txt_emb'
            )
            img_embedding_path = os.path.join(
                self.data_dir, 'clip_embedding', f'{key}.img_emb'
            )
            text_clip_embedding = self.read_bytes_embedding(text_embedding_path, np.float32)
            img_clip_embedding = self.read_bytes_embedding(img_embedding_path, np.float32)
            out['text_clip_embedding'] = torch.from_numpy(text_clip_embedding.copy())
            out['img_clip_embedding'] = torch.from_numpy(img_clip_embedding.copy())
            
        if self.return_t5_embedding:
            t5_embedding_path = os.path.join(
                self.data_dir, 't5_embedding', f'{key}.t5_emb'
            )
            t5_embedding = self.read_bytes_embedding(t5_embedding_path, np.float32).reshape(-1, 1024)
            out['t5_embedding'] = torch.from_numpy(t5_embedding.copy())
        
        return out