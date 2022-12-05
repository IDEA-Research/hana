# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-06-22 19:21:57


import os
import random
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json

from .tsv_io import TSVFile
from .combine_bin_file import CombineBinFile
from .io_common import img_from_base64

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

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
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class CLIPT5EmbeddingDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, t5_bin_file, t5_map_file, clip_bin_file, clip_map_file, **kwargs):
        """
        Args:
            tsv_file (str): path to tsv file
            t5_bin_file (str): path to t5 bin file
            t5_map_file (str): path to t5 map file
            clip_bin_file (str): path to clip embedding bin file
            clip_map_file (str): path to clip embedding map file
            kwargs: if 'tokenizer' is in kwargs, use it to tokenize text
                    if 'text_encoder' is in kwargs, use text-encoder to tokenize and encode text. 
                    currently we use pre-preocessed text-encodings result.
                    If 'text_encoder' is provided, 'tokenizer' is ignored.
        """
        self.tsv_file = tsv_file
        self.t5_bin_file, self.t5_map_file, self.clip_bin_file, self.clip_map_file = t5_bin_file, t5_map_file, clip_bin_file, clip_map_file
        self.tsv = TSVFile(tsv_file)
        self.bin = CombineBinFile(t5_bin_file, t5_map_file, clip_bin_file, clip_map_file)

        self.tokenizer = kwargs.get('tokenizer', None)
        self.transforms = _transform(224) if 'transforms' not in kwargs else kwargs['transforms']
        self.text_encoder = kwargs.get('text_encoder', None)
        if self.text_encoder != 'T5':
            assert not (self.tokenizer is not None and self.text_encoder is not None), \
                    "tokenizer and text_encoder cannot be both provided"
        self.num_rows = self.bin.num_rows()
        self.close()

    def close(self):
        self.tsv.close()
        self.bin.close()
        self.tsv = None
        self.bin = None

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
            return label['objects'][0]['caption']

    def __getitem__(self, index):
        """
        Note:
            if self.text_encoder exists, return ([clip-]image embedding, [clip-]text embedding, img, text, text-encodigs)
            elif self.tokenizer exists, return ([clip-]image embedding, [clip-]text embedding, img, text, token, mask)
            else return ([clip-]image embedding, [clip-]text embedding, img, text)
        """
        if self.tsv is None or self.bin is None:
            self.tsv = TSVFile(self.tsv_file)
            self.bin = CombineBinFile(self.t5_bin_file, self.t5_map_file, self.clip_bin_file, self.clip_map_file)
        t5_embedding, clip_embedding, idx = self.bin.seek(index)
        t5_embedding = t5_embedding.reshape(-1, 1024)
        t5_embedding = torch.tensor(t5_embedding)

        split = clip_embedding.size // 2
        clip_image_embedding = torch.tensor(clip_embedding[:split])
        clip_text_embedding = torch.tensor(clip_embedding[split:])

        row = self.tsv.seek(idx)        
        img = self.read_img(row)
        img = self.transforms(img)
        text = self.read_label(row)
        
        if self.text_encoder is not None:
            encoding = self.tokenizer(text,
                            padding='max_length',
                            max_length=256,
                            truncation=True,
                            return_tensors="pt")

            t5_text_token, t5_text_mask = encoding.input_ids[0], encoding.attention_mask[0]
            max_length = t5_text_token.shape[0]
            text_length = t5_embedding.shape[0]
            # pad t5_encodings with zero.
            t5_text_encodings = torch.cat([t5_embedding, torch.zeros(t5_embedding[-1:].shape).repeat(max_length - text_length, 1)], dim=0)
            return clip_image_embedding, clip_text_embedding, img, text, t5_text_encodings
        
        if self.tokenizer is not None:
            token, mask = self.tokenizer(text)
            token, mask = torch.tensor(token), torch.tensor(mask)
            return clip_image_embedding, clip_text_embedding, img, text, token, mask

        else:
            return clip_image_embedding, clip_text_embedding, img, text

    def __len__(self):
        return self.num_rows


def get_yaml_datasets(yaml_file_path, **kwargs):
    with open(yaml_file_path) as yaml_file:
        yaml_docs = yaml.safe_load_all(yaml_file)

        dataset_list = []
        for doc in yaml_docs:
            if doc is not None and 'dataset' in doc:
                dataset = doc['dataset']
                if dataset['data_file_type'] == "YamlDataset":
                    dataset_list += get_yaml_datasets(dataset['data_file'], **kwargs)
                else:
                    tsv_file = dataset['data_file']
                    clip_bin_file = dataset['clip_bin']
                    clip_map_file = clip_bin_file.replace(".bin", ".txt")
                    t5_bin_file = dataset['t5_bin']
                    t5_map_file = t5_bin_file.replace(".bin", ".txt")

                    dataset_list.append(CLIPT5EmbeddingDataset(tsv_file, t5_bin_file, t5_map_file, clip_bin_file, clip_map_file, **kwargs))

            # break
        return dataset_list
    

class LAIONT5CLIPEmbeddingDataset(Dataset):
    def __init__(self, yaml_file, seed=10, **kwargs):
        """
        Args:
            yaml_file (str): path to store all yaml files
            clip_bin_dir (str): path to store all clip bin files
            t5_bin_dir (str): path to store all t5 bin files
            kwargs: if 'tokenizer' is in kwargs, use it to tokenize text
                    if 'text_encoder' is in kwargs, use text-encoder to tokenize and encode text. 
                    currently we use pre-preocessed text-encodings result.
                    If 'text_encoder' is provided, 'tokenizer' is ignored.
        """
        self.datasets = get_yaml_datasets(yaml_file, **kwargs)
        self.shuffle_tsv(seed)
        
        num_rows = []
        s = 0
        for d in self.datasets:
            s += len(d)
            num_rows.append(s)

        self.num_rows = np.array(num_rows)

        self.pid_map = {}
        # self.close_all()
    
    def shuffle_tsv(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(self.datasets), generator=g).tolist()
        self.datasets = [self.datasets[i] for i in indices]
    
    def close_all(self):
        for d in self.datasets:
            d.close()
    
    def find_index(self, index):
        subset_index = np.searchsorted(self.num_rows, index, side='right')
        subset_data_index = (index - self.num_rows[subset_index-1] ) if ( subset_index > 0 ) else index
        return subset_index, subset_data_index

    def get_tsv(self, rank, num_replicas, total_size):
        assert total_size % num_replicas == 0
        num_samples = total_size // num_replicas

        start, end = rank * num_samples, (rank + 1) * num_samples
        d_s, d_e = self.find_index(start)[0], self.find_index(end)[0]

        tsv_file_list = []
        for i in range(d_s, d_e + 1):
            tsv_file_list.append(self.datasets[i].tsv_file)
        return tsv_file_list

    def get_item(self, index):
        subset_index, subset_data_index = self.find_index(index)

        if os.getpid() in self.pid_map:
            if self.pid_map[os.getpid()] != subset_index:
                self.datasets[self.pid_map[os.getpid()]].close()
                self.pid_map[os.getpid()] = subset_index
        else:
            self.pid_map[os.getpid()] = subset_index
        
        return self.datasets[subset_index][subset_data_index]
        
    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            new_index = (index + 1) % len(self)
            return self.get_item(new_index)

    def __len__(self):
        return self.num_rows[-1]