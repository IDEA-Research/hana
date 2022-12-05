# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2022-05-23 14:55:29
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2022-05-24 17:41:10
import os
import time

import torch
import clip
from PIL import Image
import numpy as np
from dataset.tsv_dataset import TSVDataset
from dataset.embedding_dataset import EmbeddingDataset

def test_embedding_bin():
    with open("/comp_robot/cv_public_dataset/ConceptualCaptions/embedding.bin", 'rb') as bf, open("/comp_robot/cv_public_dataset/ConceptualCaptions/embedding.txt", 'r') as f:
        lines = f.readlines()
        row = [line.strip().split("\t") for line in lines]

        for line in row:
            offset = int(line[0])
            lineidx = int(line[1])

            bf.seek(offset)
            data = bf.read(3072)
            array = np.frombuffer(data, dtype=np.float16)
            print(array.shape)

def write(map_file, start, index):
    with open(map_file, 'a') as f:
        f.write(f"{start}\t{index}\n")


def test_dataset():
    tsv_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv"
    bin_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.bin"
    map_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/CLIP_ViT_L_embedding.txt"
    dataset = EmbeddingDataset(tsv_file, bin_file, map_file)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=16)

    start = time.time()
    time_count = 0
    data_count = 0
    for image_embedding, text_embedding, img, text in train_loader:
        data_count += 512
        time_count = time.time() - start
        print("load data {} per second".format(data_count // time_count))

def run_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device) # ViT-L/14, ViT-L/14@336px

    tsv_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv"
    bin_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/embedding.bin"
    map_file = "/comp_robot/cv_public_dataset/ConceptualCaptions/embedding.txt"

    dataset = TSVDataset(tsv_file, transforms=preprocess)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=12)


    with torch.no_grad(), open(bin_file, 'ab') as bf:
        for image, text, meta, index in train_loader:
            image = image.to(device)
            text = text.to(device)

            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            for i, idx in enumerate(index):
                idx = idx.item()

                features = torch.cat([image_features[i], text_features[i]], dim=0).cpu().numpy()

                pos = bf.tell()
                bf.write(features.tobytes())
                write(map_file, pos, idx)


if __name__ == '__main__':
    test_dataset()

    