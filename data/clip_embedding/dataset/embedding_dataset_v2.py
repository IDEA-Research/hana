# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-09-16 11:58:33

import torch
from torch.utils.data import Dataset
import clip
import json

from .tsv_io import TSVFile
from .combine_bin_file import CombineBinFile
from .io_common import img_from_base64

class EmbeddingDatasetV2(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, clip_bin_file, clip_map_file, t5_bin_file, t5_map_file, 
                 tokenizer, transforms, **kwargs):
        self.tsv = TSVFile(tsv_file)
        self.bin = CombineBinFile(t5_bin_file, t5_map_file, clip_bin_file, clip_map_file)

        self.tokenizer = tokenizer
        self.transforms = transforms

    def close(self):
        self.tsv.close()

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
        # embedding, idx = self.bin.seek(index)

        t5_embedding, clip_embedding, idx = self.bin.seek(index)

        t5_embedding = t5_embedding.reshape(-1, 1024)
        t5_embedding = torch.tensor(t5_embedding)

        split = clip_embedding.size // 2
        clip_image_embedding = torch.tensor(clip_embedding[:split])
        clip_text_embedding = torch.tensor(clip_embedding[split:])

        row = self.tsv.seek(idx)        
        encoding = self.tokenizer(self.read_label(row),
                        padding='max_length',
                        max_length=256,
                        truncation=True,
                        return_tensors="pt")

        t5_text_token, t5_text_mask = encoding.input_ids[0], encoding.attention_mask[0]
        max_length = t5_text_token.shape[0]
        text_length = t5_embedding.shape[0]
        t5_text_embedding = torch.cat([t5_embedding, t5_embedding[-1:].repeat(max_length - text_length, 1)], dim=0)

        img = self.read_img(row)
        img = self.transforms(img)
        clip_text_token = clip.tokenize(self.read_label(row), truncate=True).squeeze()
        assert sum(t5_text_mask).item() == t5_embedding.shape[0]

        return clip_image_embedding, clip_text_embedding, img, clip_text_token, t5_text_embedding, t5_text_token, t5_text_mask

    def __len__(self):
        return self.bin.num_rows()
