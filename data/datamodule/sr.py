# coding=utf-8
# Copyright (c) 2022 IDEA. All rights reserved.
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

import json
import os
import PIL

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data.dataset import SRDataset, SRImageProcessor

INTERPOLATION = {
    "linear": PIL.Image.LINEAR,
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
}
        
class SRDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        mapping_file:str,
        return_clip_embedding:bool,
        return_t5_embedding:bool,
        image_size:int,
        val_image_size:int,
        batch_size:int,
        val_batch_size:int,
        num_workers:int, 
        interpolation:str,
        downscale_factor:int,
        min_crop_factor,
        max_crop_factor,
        random_crop:bool,
        degradation:str,
        gaussian_blur:bool,
        blur_prob:float=0.5,
        random_flip_prob:float=0.5,
    ):
        super().__init__()
        # build train mapping file and val mapping file
        self.train_mapping_file, self.val_mapping_file = self._build_mapping_file(mapping_file, val_batch_size)
        # hparam
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.return_clip_embedding = return_clip_embedding
        self.return_t5_embedding = return_t5_embedding
        
        # transforms functions
        train_HR_transforms = T.Compose([
            T.Resize((image_size, image_size), interpolation=INTERPOLATION[interpolation]),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
        ])
        train_LR_transforms = T.Compose([
            T.Resize(
                (image_size//downscale_factor, image_size//downscale_factor), 
                interpolation=INTERPOLATION[interpolation]),
            T.CenterCrop((image_size//downscale_factor, image_size//downscale_factor)),
            T.ToTensor(),
        ])
        val_HR_transforms = T.Compose([
            T.Resize((val_image_size, val_image_size), interpolation=INTERPOLATION[interpolation]),
            T.CenterCrop((val_image_size, val_image_size)),
            T.ToTensor(),
        ])
        val_LR_transforms = T.Compose([
            T.Resize(
                (val_image_size//downscale_factor, val_image_size//downscale_factor), 
                interpolation=INTERPOLATION[interpolation]),
            T.CenterCrop((val_image_size//downscale_factor, val_image_size//downscale_factor)),
            T.ToTensor(),
        ])
        
        self.sr_train_image_processor = SRImageProcessor(
            image_size=image_size,
            downscale_factor=downscale_factor,
            min_crop_factor=min_crop_factor,
            max_crop_factor=max_crop_factor,
            random_crop=random_crop,
            degradation=degradation,
            gaussian_blur=gaussian_blur,
            blur_prob=blur_prob,
            random_flip_prob=random_flip_prob,
            HR_transforms=train_HR_transforms,
            LR_transforms=train_LR_transforms,
        )
        
        self.sr_val_image_processor = SRImageProcessor(
            image_size=val_image_size,
            downscale_factor=downscale_factor,
            min_crop_factor=min_crop_factor,
            max_crop_factor=0.8,
            random_crop=random_crop,
            degradation=degradation,
            gaussian_blur=gaussian_blur,
            blur_prob=blur_prob,
            random_flip_prob=random_flip_prob,
            HR_transforms=val_HR_transforms,
            LR_transforms=val_LR_transforms,
        )
    
    def _build_mapping_file(self, mapping_file, val_batch_size):
        train_mapping_file = mapping_file.replace('.json', '_train.json')
        val_mapping_file = mapping_file.replace('.json', '_val.json')
        if os.path.exists(train_mapping_file) and os.path.exists(val_mapping_file):
            return train_mapping_file, val_mapping_file
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        keys = sorted(list(mapping.keys()))
        val_mapping = {k: mapping[k] for k in keys[:val_batch_size]}
        train_mapping = {k: mapping[k] for k in keys[val_batch_size:]}
        # dump to json file
        with open(train_mapping_file, 'w') as f:
            json.dump(train_mapping, f)
        with open(val_mapping_file, 'w') as f:
            json.dump(val_mapping, f)
        return train_mapping_file, val_mapping_file

    def setup(self, stage=None):
        self.train_set = SRDataset(
            mapping_file=self.train_mapping_file,
            sr_image_processor=self.sr_train_image_processor,
            return_clip_embedding=self.return_clip_embedding,
            return_t5_embedding=self.return_t5_embedding,
        )
        self.val_set = SRDataset(
            mapping_file=self.val_mapping_file,
            sr_image_processor=self.sr_train_image_processor,
            return_clip_embedding=self.return_clip_embedding,
            return_t5_embedding=self.return_t5_embedding,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            self.val_batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
        )