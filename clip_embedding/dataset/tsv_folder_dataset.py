# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-25 14:12:24
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-05-29 14:23:13
import os
from tqdm import tqdm
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import clip
import json
import glob

from .tsv_io import TSVFile
from .io_common import img_from_base64, generate_lineidx, FileProgressingbar

class TSVFolderDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, tsv_folder, partition=-1, transforms=None, **kwargs):
        self.tsv_list, self.num_rows = self.search_folder(tsv_folder, partition)
        self.transforms = transforms

    def search_folder(self, tsv_folder, partition):
        tsv_file_list = glob.glob(os.path.join(tsv_folder, "*.tsv"))
        tsv_file_list.sort()

        if partition >= 0:
            tsv_len = len(tsv_file_list)
            split_size = tsv_len // 32
            tsv_file_list = [tsv_file_list[i:i+split_size] for i in range(0, tsv_len, split_size)][partition]

        tsv_list = []
        num_rows = []
        s = 0
        for tsv in tqdm(tsv_file_list):
            tsv_file = TSVFile(tsv)
            tsv_list.append(tsv_file)
            s += tsv_file.num_rows()
            num_rows.append(s)
        return tsv_list, num_rows

    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_label(self, row):
        label = json.loads(row[1])
        return label['objects'][0]['caption']

    def find_index(self, index):
        subset_index = np.searchsorted(self.num_rows, index, side='right')
        subset_data_index = (index - self.num_rows[subset_index-1] ) if ( subset_index > 0 ) else index
        return subset_index, subset_data_index

    def __getitem__(self, index):
        tsv_index, tsv_offset = self.find_index(index)
        row = self.tsv_list[tsv_index].seek(tsv_offset)
        try:
            img = self.read_img(row)
            text = clip.tokenize(self.read_label(row), truncate=True).squeeze()
        except Exception as e:
            print("error img: {}".format(index))
            index = -1
            # tsv_index, tsv_offset = self.find_index(index)
            row = self.tsv_list[0].seek(0)
            img = self.read_img(row)
            text = clip.tokenize(self.read_label(row), truncate=True).squeeze()

        if self.transforms:
            img = self.transforms(img)

        meta = row[1]
        return img, text, meta, index

    def __len__(self):
        return self.num_rows[-1]
