# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-06-21 17:09:45


import os

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import clip
import json

from .tsv_io import TSVFile
from .bin_file import BinFile
from .io_common import img_from_base64

from transformers import T5Tokenizer


class T5EmbeddingDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, bin_file, map_file, **kwargs):
        self.tsv = TSVFile(tsv_file)
        self.bin = BinFile(bin_file, map_file)

        self.tokenizer = T5Tokenizer.from_pretrained("t5-11b")

    def close(self):
        self.tsv.close()

    def read_label(self, row):
        label = json.loads(row[1])

        if 'text' in label:
            return label['text']
        elif 'caption' in label:
            return label['caption']
        else:
            return label['objects'][0]['caption']

    def __getitem__(self, index):
        embedding, idx = self.bin.seek(index)
        embedding = embedding.reshape(-1, 1024)
        embedding = torch.tensor(embedding)

        row = self.tsv.seek(idx)        
        encoding = self.tokenizer(self.read_label(row),
                        padding='max_length',
                        max_length=256,
                        truncation=True,
                        return_tensors="pt")

        input_ids, attention_mask = encoding.input_ids[0], encoding.attention_mask[0]

        max_length = input_ids.shape[0]
        text_length = embedding.shape[0]
        embedding = torch.cat([embedding, embedding[-1:].repeat(max_length - text_length, 1)], dim=0)
        return embedding, input_ids, attention_mask

    def __len__(self):
        return self.bin.num_rows()