# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 15:28:49
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-05-23 16:48:52

from torch.utils.data import Dataset
import clip
import json

from .tsv_io import TSVFile
from .io_common import img_from_base64

class TSVDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_file, repeat_time=1, transforms=None, **kwargs):
        self.tsv = TSVFile(tsv_file)
        self.repeat_time = repeat_time

        self.transforms = transforms

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
            raise ValueError("No text in label")

    def __getitem__(self, index):
        index = index % self.real_len
        row = self.tsv.seek(index)
        try:
            img = self.read_img(row)
        except Exception as e:
            print("error img: {}".format(index))
            index = index + 1
            row = self.tsv.seek(index)
            img = self.read_img(row)

        if self.transforms:
            img = self.transforms(img)

        text = clip.tokenize(self.read_label(row), truncate=True).squeeze()
        meta = row[1]
        return img, text, meta, index

    def __len__(self):
        return int(self.tsv.num_rows() * self.repeat_time)


