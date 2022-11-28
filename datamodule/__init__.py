from .multi_dataset import MultiEmbedDataModule, MultiSRDataModule
import torchvision.transforms as T
import PIL
from utils.tokenizer import get_encoder
from transformers import T5Tokenizer


def get_datamodule(config):
    size = config.image_size
    text_ctx = config.text_ctx
    interpolation = {"linear": PIL.Image.LINEAR,
                     "bilinear": PIL.Image.BILINEAR,
                     "bicubic": PIL.Image.BICUBIC,
                     "lanczos": PIL.Image.LANCZOS,
                     }[config.interpolation]

    # set up image transforms
    train_transforms = T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        T.RandomHorizontalFlip(config.p_flip),
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    # set tokenizer
    tokenizer = get_encoder(text_ctx=text_ctx) if config.use_tokenizer else None
    text_encoder = None
    if config.use_pretrained_text_encoder:
        text_encoder = 'T5'
        tokenizer = T5Tokenizer.from_pretrained("t5-11b")

    return MultiEmbedDataModule(
        config=config,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )


def get_SRdatamodule(config):
    text_ctx = config.text_ctx
    tokenizer = get_encoder(text_ctx=text_ctx) if config.use_tokenizer else None
    text_encoder = None
    if config.use_pretrained_text_encoder:
        text_encoder = 'T5'
        tokenizer = T5Tokenizer.from_pretrained("t5-11b")

    size = config.image_size
    lr_size = size // config.downscale_factor
    val_size = config.test_image_size
    val_lr_size = val_size // config.downscale_factor

    interpolation = {"linear": PIL.Image.LINEAR,
                     "bilinear": PIL.Image.BILINEAR,
                     "bicubic": PIL.Image.BICUBIC,
                     "lanczos": PIL.Image.LANCZOS,
                     }[config.degradation]

    # set up image transforms
    train_HR_transforms = T.Compose([
        T.Resize(size, interpolation=interpolation),
        T.CenterCrop(size),
        T.ToTensor(),
    ])
    train_LR_transforms = T.Compose([
        T.Resize(lr_size, interpolation=interpolation),
        T.CenterCrop(lr_size),
        T.ToTensor(),
    ])
    val_HR_transforms = T.Compose([
        T.Resize(val_size, interpolation=interpolation),
        T.CenterCrop(val_size),
        T.ToTensor(),
    ])
    val_LR_transforms = T.Compose([
        T.Resize(val_lr_size, interpolation=interpolation),
        T.CenterCrop(val_lr_size),
        T.ToTensor(),
    ])
    return MultiSRDataModule(
        config=config,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_HR_transforms=train_HR_transforms,
        train_LR_transforms=train_LR_transforms,
        val_HR_transforms=val_HR_transforms,
        val_LR_transforms=val_LR_transforms,
    )


class DummyTextEncoder(object):
    def __init__(self, text_ctx, text_emb_dim):
        self.text_ctx = text_ctx
        self.text_emb_dim = text_emb_dim

    def encode(self, text):
        import torch
        return torch.randn(self.text_ctx, self.text_emb_dim)
