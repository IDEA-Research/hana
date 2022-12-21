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

import os
import PIL
import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from data.dataset import BaseDataset

INTERPOLATION = {
    "linear": PIL.Image.LINEAR,
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
}

class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        mapping_file:str,
        return_clip_embedding:bool,
        return_t5_embedding:bool,
        image_size:int,
        batch_size:int,
        val_batch_size:int,
        num_workers:int, 
        p_flip:float=0.5,
        interpolation:str='bilinear',
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
        self.train_transforms = T.Compose([
            T.Resize((image_size, image_size), interpolation=INTERPOLATION[interpolation]),
            T.CenterCrop((image_size, image_size)),
            T.RandomHorizontalFlip(p_flip),
            T.ToTensor(),
        ])
        self.val_transforms = T.Compose([
            T.Resize((image_size, image_size), interpolation=INTERPOLATION[interpolation]),
            T.CenterCrop((image_size, image_size)),
            T.ToTensor(),
        ])
        
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
        self.train_set = BaseDataset(
            mapping_file=self.train_mapping_file, transforms=self.train_transforms,
            return_clip_embedding=self.return_clip_embedding,
            return_t5_embedding=self.return_t5_embedding
        )
        self.val_set = BaseDataset(
            mapping_file=self.val_mapping_file, transforms=self.val_transforms,
            return_clip_embedding=self.return_clip_embedding,
            return_t5_embedding=self.return_t5_embedding
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