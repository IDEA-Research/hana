# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-07-26 13:48:36
import os
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import clip
import json
import glob
import yaml

from transformers import T5Tokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .tsv_io import TSVFile
from .bin_file import BinFile
from .io_common import img_from_base64
from .embedding_dataset_v2 import EmbeddingDatasetV2

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_yaml_datasets(yaml_file_path, clip_bin_dir, t5_bin_dir, tokenizer, transforms):
    with open(yaml_file_path) as yaml_file:
        yaml_docs = yaml.safe_load_all(yaml_file)

        dataset_list = []
        for doc in yaml_docs:
            if doc is not None and 'dataset' in doc:
                dataset = doc['dataset']
                if dataset['data_file_type'] == "YamlDataset":
                    dataset_list += get_yaml_datasets(
                        dataset['data_file'], clip_bin_dir, t5_bin_dir, tokenizer, transforms)
                else:
                    tsv_file = dataset['data_file']
                    clip_bin_file = os.path.join(
                        clip_bin_dir, os.path.basename(tsv_file).replace(".tsv", ".bin"))
                    clip_map_file = os.path.join(
                        clip_bin_dir, os.path.basename(tsv_file).replace(".tsv", ".txt"))
                    t5_bin_file = os.path.join(t5_bin_dir, os.path.basename(
                        tsv_file).replace(".tsv", ".bin"))
                    t5_map_file = os.path.join(t5_bin_dir, os.path.basename(
                        tsv_file).replace(".tsv", ".txt"))

                    dataset_list.append(EmbeddingDatasetV2(
                        tsv_file, clip_bin_file, clip_map_file, t5_bin_file, t5_map_file, tokenizer, transforms))

            break
        return dataset_list

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

class EmbeddingTSVDatasetV2(Dataset):
    '''
        TSV dataset for tsv file
    '''

    def __init__(self, yaml_file, clip_bin_dir=None, t5_bin_dir=None, seed=10, **kwargs):
        tokenizer = T5Tokenizer.from_pretrained("t5-11b")
        transforms = _transform(224)
        self.datasets = get_yaml_datasets(yaml_file, clip_bin_dir, t5_bin_dir, tokenizer, transforms)
        
        self.shuffle_tsv(seed)

        num_rows = []
        s = 0
        for d in self.datasets:
            s += len(d)
            num_rows.append(s)

        self.num_rows = np.array(num_rows)

        self.pid_map = {}
        self.close_all()

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
        subset_data_index = (
            index - self.num_rows[subset_index-1]) if (subset_index > 0) else index
        return subset_index, subset_data_index

    def get_tsv(self, rank, num_replicas, total_size, target_path):
        assert total_size % num_replicas == 0
        num_samples = total_size // num_replicas

        start, end = rank * num_samples, (rank + 1) * num_samples
        d_s, d_e = self.find_index(start)[0], self.find_index(end)[0]

        tsv_file_list = []
        for i in range(d_s, d_e + 1):
            tsv_file_list.append(self.datasets[i].tsv.tsv_file)
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
