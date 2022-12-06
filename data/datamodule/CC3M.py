'''
Description: 
version: 
Author: ciao
Date: 2022-05-25 01:35:36
LastEditTime: 2022-05-27 01:03:20
'''
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from mm_data import get_dataset as get_mm_dataset
from clip_embedding import get_dataset as get_embed_dataset
from clip_embedding import get_SR_dataset


class CC3MDataModule(pl.LightningDataModule):
    def __init__(self, config, train_transforms,
                 val_transforms=None, tokenizer=None):
        """CC3M Dataset LightningDataModule

        Args:
            batch_size (int): train batch size
            val_batch_size (int): validation batch size
            train_transforms (Composed transforms | dict)
            val_transforms (Composed transforms | dic): if None, use simple 'ToTensor'
            tokenizer (obj): tokenizer for text, callable object
        """
        super().__init__()
        self.config = config
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms if val_transforms else [
            dict(type='ToTensor')]
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        self.train_set = get_mm_dataset(
            'CC_3M', split='train', image_transforms=self.train_transforms,
            tokenizer=self.tokenizer,
        )

        self.val_set = get_mm_dataset(
            'CC_3M', split='val', image_transforms=self.val_transforms,
            tokenizer=self.tokenizer,
        )
        self.val_set = Subset(self.val_set, range(
            self.config.val_batch_size))  # only fetch first batch, TODO: val-lineidx

    def train_dataloader(self):
        return DataLoader(
            self.train_set, self.config.batch_size, 
            shuffle=True, num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.config.val_batch_size, 
            shuffle=False, num_workers=self.config.num_workers,
        )


class CC3MEmbedDataModule(pl.LightningDataModule):
    def __init__(self, config, train_transforms, val_transforms=None,
                 tokenizer=None, text_encoder=None):
        super().__init__()
        self.config = config
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms if val_transforms else [
            dict(type='ToTensor')]
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def setup(self, stage=None):
        self.train_set = get_embed_dataset(
            'CC_3M', split='train', image_transforms=self.train_transforms,
            tokenizer=self.tokenizer, text_encoder=self.text_encoder,
        )

        self.val_set = get_embed_dataset(
            'CC_3M', split='val', image_transforms=self.val_transforms,
            tokenizer=self.tokenizer, text_encoder=self.text_encoder,
        )
        
        self.val_set = Subset(self.val_set, range(
            self.config.val_batch_size))  # only fetch first batch, TODO: val-lineidx

    def train_dataloader(self):
        return DataLoader(
            self.train_set, self.config.batch_size, 
            shuffle=True, num_workers=self.config.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.config.batch_size, 
            shuffle=False, num_workers=self.config.num_workers,
        )


class CC3MSRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size

    def setup(self, stage=None):
        self.train_set = get_SR_dataset(
            'CC_3M', split='train', config=self.config,
        )
        self.val_set = get_SR_dataset(
            'CC_3M', split='val', config=self.config,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, self.batch_size, shuffle=True, num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.val_batch_size, shuffle=False, num_workers=4,
        )
