# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-06-19 18:35:15


import os

import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import clip
import json
from transformers import T5Tokenizer

from .tsv_io import TSVFile
from .io_common import img_from_base64, generate_lineidx, FileProgressingbar

class T5TSVDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, **kwargs):
        self.tsv = TSVFile(tsv_file)

        self.tokenizer = T5Tokenizer.from_pretrained("t5-11b")

    @property
    def real_len(self):
        return self.tsv.num_rows()

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
        index = index % self.real_len
        row = self.tsv.seek(index)
        try:
            encoding = self.tokenizer(self.read_label(row),
                        padding='max_length',
                        max_length=256,
                        truncation=True,
                        return_tensors="pt")

        except Exception as e:
            print("error img: {}".format(index))
            index = -1
            # tsv_index, tsv_offset = self.find_index(index)
            row = self.tsv.seek(0)
            encoding = self.tokenizer(self.read_label(row),
                        padding='max_length',
                        max_length=256,
                        truncation=True,
                        return_tensors="pt")
        

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        meta = row[1]
        return input_ids, attention_mask, meta, index

    def __len__(self):
        return self.tsv.num_rows()



