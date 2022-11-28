"""
    Test datasets.py
"""

import argparse
from torch.utils.data import DataLoader, ConcatDataset
import datasets
from mm_data.datasets import Dataset
from mm_data.datasets import Split

parser = argparse.ArgumentParser()
parser.add_argument('--workers_per_gpu', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--imgs_per_gpu', default=2, type=int,
                    help='batch size per gpu')

args = parser.parse_args()


image_transforms = [
    dict(type='RandomResizedCrop', size=256, scale=(0.75, 1.), ratio=(1., 1.)),
    dict(type='ToTensor'),
]

dataset_list = []
for test_ds in [Dataset.YFCC, Dataset.IN_22K, Dataset.IN_1K, Dataset.COCO_17]:
    # Loading 40m CLICKTURE is time-consuming, skipping here
    dataset, tsv_file = datasets.get_dataset(test_ds, image_transforms=image_transforms)
    print(f'{test_ds}/: {len(dataset)}')
    dataset_list.append(dataset)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.imgs_per_gpu,
        num_workers=args.workers_per_gpu,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2)


for split in [Split.ALL, Split.TRAIN, Split.TEST]:
    dataset, tsv_file = datasets.get_dataset(Dataset.CC_3M, split=split, image_transforms=image_transforms)
    print(f'{datasets.Dataset.CC_3M}/{split}: {len(dataset)}')
    if split == Split.ALL:
        dataset_list.append(dataset)
    data_loader = DataLoader(
        dataset, 
        batch_size=args.imgs_per_gpu,
        num_workers=args.workers_per_gpu,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2)


monster_dataset = ConcatDataset(dataset_list)
print(f'Combined Dataset Size: {len(monster_dataset)}')
monster_dataloader =  DataLoader(
        monster_dataset, 
        batch_size=args.imgs_per_gpu,
        num_workers=args.workers_per_gpu,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2)

# see how the data looks like.
for i, (images, info) in enumerate(monster_dataloader):
    print(images.shape, info)
    if i > 0:
        break