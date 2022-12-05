import copy
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset

from clip_embedding import get_dataset as get_embed_dataset
from clip_embedding import get_SR_dataset
from clip_embedding.dataset.large_scale_distributed_sampler import LargeScaleDistributedSampler
from eval import eval_ds, sr_eval_ds


class MultiEmbedDataModule(pl.LightningDataModule):
    def __init__(self, config, train_transforms, val_transforms=None,
                 tokenizer=None, text_encoder=None):
        super().__init__()
        # hparam
        self.config = config
        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.num_workers = config.num_workers
        # transforms functions
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms if val_transforms else [
            dict(type='ToTensor')]
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def setup(self, stage=None):
        dataname = self.config.dataname
        # DO NOT concat dataset, otherwise distributed sampler would fail to function properly
        # (by keep opening and closing file handlers), causing severe slowness.
        # If you wish to combine several datasets, do that via Yaml file.
        assert isinstance(dataname, str)
        total_set = get_embed_dataset(
            dataset=dataname, image_transforms=self.train_transforms,
            tokenizer=self.tokenizer, text_encoder=self.text_encoder,
            copy2local=self.config.copy2local,
        )
        val_set = Subset(total_set, range(
            0, self.val_batch_size))

        self.train_set = total_set
        self.val_set = val_set
        self.test_set = eval_ds

    def train_dataloader(self):
        dataset = self.train_set
        if not hasattr(dataset, 'datasets'):
            dataset = ConcatDataset([dataset])
        # TODO: build distributed sampler only laion400
        train_sampler = LargeScaleDistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset, 
            self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.val_batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, self.val_batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

class MultiSRDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer=None, text_encoder=None,
                 train_HR_transforms=None,
                 train_LR_transforms=None,
                 val_HR_transforms=None,
                 val_LR_transforms=None,
                 ):
        super().__init__()
        # hparam
        self.config = config
        self.batch_size = config.batch_size
        self.val_batch_size = config.val_batch_size
        self.num_workers = config.num_workers
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.train_HR_transforms = train_HR_transforms
        self.train_LR_transforms = train_LR_transforms
        self.val_HR_transforms = val_HR_transforms
        self.val_LR_transforms = val_LR_transforms

    def setup(self, stage=None):
        dataname = self.config.dataname
        # DO NOT concat dataset, otherwise distributed sampler would fail to function properly
        # (by keep opening and closing file handlers), causing severe slowness.
        # If you wish to combine several datasets, do that via Yaml file.
        assert isinstance(dataname, str)
        total_set = get_SR_dataset(
            config=self.config,
            dataset=dataname,
            tokenizer=self.tokenizer, text_encoder=self.text_encoder,
            copy2local=self.config.copy2local,
            train_HR_transforms=self.train_HR_transforms,
            train_LR_transforms=self.train_LR_transforms,
        )
        total_sample_num = len(total_set)
        val_set = Subset(total_set, range(
            total_sample_num - self.val_batch_size, total_sample_num))

        self.train_set = total_set
        self.val_set = val_set
        test_config = copy.deepcopy(self.config)
        test_config.image_size = test_config.test_image_size
        test_config.max_crop_factor = 0.8
        self.test_set = sr_eval_ds(config=test_config,
                                   HR_transforms=self.val_HR_transforms, 
                                   LR_transforms=self.val_LR_transforms)

    def train_dataloader(self):
        dataset = self.train_set
        if not hasattr(dataset, 'datasets'):
            dataset = ConcatDataset([dataset])
        # TODO: build distributed sampler only laion400
        train_sampler = LargeScaleDistributedSampler(dataset, shuffle=True)
        return DataLoader(
            dataset, 
            self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.val_batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, self.val_batch_size,
            shuffle=False, num_workers=self.num_workers,
        )