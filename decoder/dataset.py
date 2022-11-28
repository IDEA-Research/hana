from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import json
import os

def get_loader(batch_size, resolution, image_paths, clip_embedings, tokens, masks, pad_token=50256, zero_clip_emb_prob=0.1, zero_text_prob=0.5, shuffle=True,):  
    dataset = ImageDataset(resolution, image_paths, clip_embedings, tokens, masks, pad_token, zero_clip_emb_prob, zero_text_prob)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    while True:
        yield from loader
        
        
def get_second_loader(batch_size, resolution, json_paths, main_dir, pad_token=50256, zero_clip_emb_prob=0.1, zero_text_prob=0.5, shuffle=True,):
    dataset = SecondImageDataset(resolution, json_paths, main_dir, pad_token, zero_clip_emb_prob, zero_text_prob)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, drop_last=True
    )
    while True:
        yield from loader
   
class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, clip_embedings, tokens, masks, pad_token=50256, zero_clip_emb_prob=0.1, zero_text_prob=0.5):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.clip_embedings = clip_embedings
        self.tokens = tokens
        self.masks = masks
        self.pad_token = pad_token
        self.zero_clip_emb_prob = zero_clip_emb_prob
        self.zero_text_prob = zero_text_prob

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}

        clip_embeding = self.clip_embedings[idx]
        if np.random.binomial(1, self.zero_clip_emb_prob):
            clip_embeding = [0] * len(clip_embeding)
        clip_embeding = torch.tensor(clip_embeding).float()

        tokens_sample = self.tokens[idx]
        mask = self.masks[idx]
        if np.random.binomial(1, self.zero_text_prob):
            tokens_sample = [self.pad_token] * len(tokens_sample)
            mask = [False] * len(mask)
        tokens_sample = torch.tensor(tokens_sample)
        mask = torch.tensor(
            mask,
            dtype=torch.bool,
            )
        out_dict["clip_emb"] = clip_embeding
        out_dict["tokens"] = tokens_sample
        out_dict["mask"] = mask
        return np.transpose(arr, [2, 0, 1]), out_dict
class SecondImageDataset(Dataset):
    def __init__(self, resolution, json_paths, main_dir, pad_token=50256, zero_clip_emb_prob=0.1, zero_text_prob=0.5):
        super().__init__()
        self.resolution = resolution
        self.main_dir = main_dir
        self.json_paths = json_paths
        self.pad_token = pad_token
        self.zero_clip_emb_prob = zero_clip_emb_prob
        self.zero_text_prob = zero_text_prob

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        
        with open(os.path.join(self.main_dir, self.json_paths[idx])) as json_file:
            in_data = json.load(json_file)
        path = in_data['path']
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}

        clip_embeding = [float(i) for i in in_data['clip_emb']]
        if np.random.binomial(1, self.zero_clip_emb_prob):
            clip_embeding = [0] * len(clip_embeding)
        clip_embeding = torch.tensor(clip_embeding).float()

        tokens_sample = in_data['tokens']
        mask = in_data['masks']
        if np.random.binomial(1, self.zero_text_prob):
            tokens_sample = [self.pad_token] * len(tokens_sample)
            mask = [False] * len(mask)
        tokens_sample = torch.tensor(tokens_sample)
        mask = torch.tensor(
            mask,
            dtype=torch.bool,
            )
        out_dict["clip_emb"] = clip_embeding
        out_dict["tokens"] = tokens_sample
        out_dict["mask"] = mask
        return np.transpose(arr, [2, 0, 1]), out_dict
