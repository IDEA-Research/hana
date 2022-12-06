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
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self, 
        mapping_file, 
        transforms=None, 
        return_clip_embedding=False, 
        return_t5_embedding=False
    ) -> None:
        super(BaseDataset, self).__init__()
        self.transforms = transforms
        self.mapping = self._load_mapping(mapping_file)
        self.keys = list(self.mapping.keys())
        self.data_dir = os.path.dirname(mapping_file)
        
        assert return_clip_embedding or return_t5_embedding, 'At least one of return_clip_embedding and return_t5_embedding should be True'
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
        if self.transforms is not None:
            img = self.transforms(img)
        caption = self.mapping[key]['caption']
        
        out = {}
        out['img'] = img
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