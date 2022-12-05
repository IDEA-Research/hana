import os
import random
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import albumentations
import cv2
from functools import partial
import PIL
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from clip_embedding.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
from .tsv_io import TSVFile
from .combine_bin_file import CombineBinFile
from .bin_file import BinFile
from .io_common import img_from_base64

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class SR_ImageProcessing(object):
    '''
        SuperResolution Image processing.

        Superresolution Dataset
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn
    '''

    def __init__(self, config, HR_transforms, LR_transforms, blur_prob=0.5, random_flip_prob=0.5):
        """
        Args:
            config:
                image_size: image size read from tsv file
                downscale_factor: Low Resolution Downscale Factor
                min_crop_factor: determines crop size s, where s = c * min_img_side_len with c 
                                sampled from interval (min_crop_f, max_crop_f)
                max_crop_factor: see above
                random_crop: bool, enable random crop if True, else center crop
                degradation: str, ["bsrgan", "bsrgan_light", interpolation_mode etc]

        """
        self.config = config
        self.HR_transforms = HR_transforms
        self.LR_transforms = LR_transforms
        self.blur_prob = blur_prob
        self.flip_prob = random_flip_prob

        if self.config.gaussian_blur:
            self.blur = T.transforms.GaussianBlur(3, 0.6)
        self.flip = T.RandomHorizontalFlip(p=1.)

        self.use_album = self.HR_transforms is None or self.LR_transforms is None
        if self.use_album:
            self._init_bsrgan_degrade(config)


    def _init_bsrgan_degrade(self, config):
        self.size = config.image_size
        self.downscale_f = config.downscale_factor
        self.LR_size = self.size // self.downscale_f
        self.min_crop_f = config.min_crop_factor
        self.max_crop_f = config.max_crop_factor
        assert 0.0 <= self.min_crop_f <= self.max_crop_f <= 1.0, \
            "min_crop_f and max_crop_f must be between 0 and 1"
        self.center_crop = not config.random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=self.size, interpolation=cv2.INTER_AREA)

        # gets reset later if incase interp_op is from pillow
        self.pil_interpolation = False

        if config.degradation == "bsrgan":
            self.degradation_process = partial(
                degradation_fn_bsr, sf=self.downscale_f)
        elif config.degradation == "bsrgan_light":
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=self.downscale_f)
        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[config.degradation]
            self.pil_interpolation = config.degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize, size=[self.LR_size,self.LR_size], interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)

    def image_transform(self, img):
        if not self.use_album:
            HR_img = self.HR_transforms(img)
            LR_img = self.LR_transforms(img)
            # random flip
            if random.random() < self.flip_prob:
                HR_img = self.flip(HR_img)
                LR_img = self.flip(LR_img)
            # apply gaussian blur noise
            if self.config.gaussian_blur and random.random() < self.blur_prob:
                LR_img = self.blur(LR_img)
            return HR_img, LR_img
        else:
            img = np.array(img).astype(np.uint8)

            # crop
            min_side_len = min(img.shape[:2])
            crop_side_len = min_side_len * \
                np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = max(int(crop_side_len), 1)

            if self.center_crop:
                self.cropper = albumentations.CenterCrop(
                    height=crop_side_len, width=crop_side_len)
            else:
                self.cropper = albumentations.RandomCrop(
                    height=crop_side_len, width=crop_side_len)
            
            if self.config.flip is not None:
                img = albumentations.Flip(p=self.config.flip)(image=img)["image"]

            img = self.cropper(image=img)['image']
            img = self.image_rescaler(image=img)['image']

            # downsample interpolation
            if self.pil_interpolation:
                img_pil = PIL.Image.fromarray(img)
                LR_img = self.degradation_process(img_pil)
                LR_img = np.array(LR_img).astype(np.uint8)
            else:
                LR_img = self.degradation_process(image=img)['image']
            
            # apply gaussian blur noise
            if self.config.gaussian_blur:
                LR_img = albumentations.GaussianBlur(
                    blur_limit=(3,3), sigma_limit=(0.6,0.6)
                )(image=LR_img)['image']

            img, LR_img = map(TF.to_tensor, [img, LR_img])
            return img, LR_img


class SRDataset(Dataset):
    '''
        TSV dataset for tsv file
    '''    
    def __init__(self, config, tsv_file, t5_bin_file, t5_map_file, clip_bin_file, clip_map_file,
                 train_HR_transforms, train_LR_transforms, **kwargs):
        """
        Args:
            tsv_file (str): path to tsv file
            t5_bin_file (str): path to t5 bin file
            t5_map_file (str): path to t5 map file
            clip_bin_file (str): path to clip embedding bin file
            clip_map_file (str): path to clip embedding map file
            kwargs: if 'tokenizer' is in kwargs, use it to tokenize text
                    if 'text_encoder' is in kwargs, use text-encoder to tokenize and encode text. 
                    currently we use pre-preocessed text-encodings result.
                    If 'text_encoder' is provided, 'tokenizer' is ignored.
        """
        self.tsv_file = tsv_file
        self.tsv = TSVFile(tsv_file)
        self.t5_bin_file, self.t5_map_file, self.clip_bin_file, self.clip_map_file = t5_bin_file, t5_map_file, clip_bin_file, clip_map_file
        self.sr_img_only = config.sr_img_only
        if self.sr_img_only:
            self.bin = BinFile(clip_bin_file, clip_map_file)
        else:
            self.bin = CombineBinFile(t5_bin_file, t5_map_file, clip_bin_file, clip_map_file)

        self.tokenizer = kwargs.get('tokenizer', None)
        self.text_encoder = kwargs.get('text_encoder', None)
        if self.text_encoder != 'T5':
            assert not (self.tokenizer is not None and self.text_encoder is not None), \
                    "tokenizer and text_encoder cannot be both provided"
        self.sr_processor = SR_ImageProcessing(config, train_HR_transforms, train_LR_transforms)
        self.num_rows = self.bin.num_rows()
        self.close()

    def close(self):
        self.tsv.close()
        self.bin.close()
        self.tsv = None
        self.bin = None

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
        """
        Note:
            if self.text_encoder exists, return ([clip-]image embedding, [clip-]text embedding, img, text, text-encodigs)
            elif self.tokenizer exists, return ([clip-]image embedding, [clip-]text embedding, img, text, token, mask)
            else return ([clip-]image embedding, [clip-]text embedding, img, text)
        """
        if self.tsv is None or self.bin is None:
            self.tsv = TSVFile(self.tsv_file)
            if self.sr_img_only:
                self.bin = BinFile(self.clip_bin_file, self.clip_map_file)
            else:
                self.bin = CombineBinFile(self.t5_bin_file, self.t5_map_file, self.clip_bin_file, self.clip_map_file)

        if self.sr_img_only:
            def get_item(index):    
                *_, idx = self.bin.seek(index)
                row = self.tsv.seek(idx)              
                img = self.read_img(row)
                return row, img
            success = False
            while not success:
                row, img = get_item(index)
                if min(img.size[:2]) < 64:
                    index = (index + random.randint(0, len(self)))%len(self)
                else:
                    success = True
            img, LR_img = self.sr_processor.image_transform(img)

            # img = self.transforms(img)
            text = self.read_label(row)
            return img, LR_img, text

        else:
            t5_embedding, clip_embedding, idx = self.bin.seek(index)
            t5_embedding = t5_embedding.reshape(-1, 1024)
            t5_embedding = torch.tensor(t5_embedding)

            split = clip_embedding.size // 2
            clip_image_embedding = torch.tensor(clip_embedding[:split])
            clip_text_embedding = torch.tensor(clip_embedding[split:])

            row = self.tsv.seek(idx)        
            img = self.read_img(row)
            img, LR_img = self.sr_processor.image_transform(img)
            # img = self.transforms(img)
            text = self.read_label(row)
            
            if self.text_encoder is not None:
                encoding = self.tokenizer(text,
                                padding='max_length',
                                max_length=256,
                                truncation=True,
                                return_tensors="pt")

                t5_text_token, t5_text_mask = encoding.input_ids[0], encoding.attention_mask[0]
                max_length = t5_text_token.shape[0]
                text_length = t5_embedding.shape[0]
                # pad t5_encodings with zero.
                t5_text_encodings = torch.cat([t5_embedding, torch.zeros(t5_embedding[-1:].shape).repeat(max_length - text_length, 1)], dim=0)
                return img, LR_img, t5_text_encodings, text, clip_image_embedding, clip_text_embedding
            if self.tokenizer is not None:
                token, mask = self.tokenizer(text)
                token, mask = torch.tensor(token), torch.tensor(mask)
                return img, LR_img, text, token, mask, clip_image_embedding, clip_text_embedding

            else:
                return img, LR_img, text, clip_image_embedding, clip_text_embedding

    def __len__(self):
        return self.num_rows


def get_sr_yaml_datasets(config, yaml_file_path, train_HR_transforms, train_LR_transforms, **kwargs):
    with open(yaml_file_path) as yaml_file:
        yaml_docs = yaml.safe_load_all(yaml_file)

        dataset_list = []
        for doc in yaml_docs:
            if doc is not None and 'dataset' in doc:
                dataset = doc['dataset']
                if dataset['data_file_type'] == "YamlDataset":
                    dataset_list += get_sr_yaml_datasets(config, dataset['data_file'], **kwargs)
                else:
                    tsv_file = dataset['data_file']
                    clip_bin_file = dataset['clip_bin']
                    clip_map_file = clip_bin_file.replace(".bin", ".txt")
                    t5_bin_file = dataset['t5_bin']
                    t5_map_file = t5_bin_file.replace(".bin", ".txt")

                    dataset_list.append(SRDataset(config, tsv_file, t5_bin_file, t5_map_file, clip_bin_file, clip_map_file, 
                                                  train_HR_transforms, train_LR_transforms, **kwargs))

            # break
        return dataset_list
    

class LAIONSRDataset(Dataset):
    def __init__(self, config, yaml_file, seed=10, train_HR_transforms=None, train_LR_transforms=None, **kwargs):
        """
        Args:
            yaml_file (str): path to store all yaml files
            clip_bin_dir (str): path to store all clip bin files
            t5_bin_dir (str): path to store all t5 bin files
            kwargs: if 'tokenizer' is in kwargs, use it to tokenize text
                    if 'text_encoder' is in kwargs, use text-encoder to tokenize and encode text. 
                    currently we use pre-preocessed text-encodings result.
                    If 'text_encoder' is provided, 'tokenizer' is ignored.
        """
        self.datasets = get_sr_yaml_datasets(config, yaml_file, train_HR_transforms, train_LR_transforms, **kwargs)
        self.shuffle_tsv(seed)
        
        num_rows = []
        s = 0
        for d in self.datasets:
            s += len(d)
            num_rows.append(s)

        self.num_rows = np.array(num_rows)

        self.pid_map = {}
        # self.close_all()
    
    def shuffle_tsv(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(self.datasets), generator=g).tolist()
        self.datasets = [self.datasets[i] for i in indices]
    
    def close_all(self):
        for d in self.datasets:
            d.close()
    
    def find_index(self, index):
        subset_index = np.searchsorted(self.num_rows, index, side='right')
        subset_data_index = (index - self.num_rows[subset_index-1] ) if ( subset_index > 0 ) else index
        return subset_index, subset_data_index

    def get_tsv(self, rank, num_replicas, total_size):
        assert total_size % num_replicas == 0
        num_samples = total_size // num_replicas

        start, end = rank * num_samples, (rank + 1) * num_samples
        d_s, d_e = self.find_index(start)[0], self.find_index(end)[0]

        tsv_file_list = []
        for i in range(d_s, d_e + 1):
            tsv_file_list.append(self.datasets[i].tsv_file)
        return tsv_file_list

    def get_item(self, index):
        subset_index, subset_data_index = self.find_index(index)

        if os.getpid() in self.pid_map:
            if self.pid_map[os.getpid()] != subset_index:
                self.datasets[self.pid_map[os.getpid()]].close()
                self.pid_map[os.getpid()] = subset_index
        else:
            self.pid_map[os.getpid()] = subset_index
        
        return self.datasets[subset_index][subset_data_index]
        
    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            new_index = (index + random.randint(0, len(self))) % len(self)
            return self.get_item(new_index)

    def __len__(self):
        return self.num_rows[-1]