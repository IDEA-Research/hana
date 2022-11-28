""""
    Multi-Modal Datasets available at IDEA.
    Data is served with (image, info) pairs. info is a dictionary containing image captions and more.

    # Dataset count statistics (may contain invalid images):
        Dataset.CC_3M/Split.ALL: 2,621,526
        Dataset.CC_3M/Split.TRAIN: 2,359,373
        Dataset.CC_3M/Split.TEST: 262,153
        Dataset.CC_12M/: 10,372,953
        Dataset.YFCC/: 14,786,676
        Dataset.CLICKTURE/: 40,000,000
        Dataset.IN_22K/: 14,198,361
        Dataset.IN_1K/: 1,281,167
        Dataset.IN_1K_VAL/: 50,000
        Dataset.COCO_17/: 118287
        Dataset.COCO_17_VAL/: 5000
"""

import enum
from mm_data.mm_tsv_dataset import MMTSVDataset, MMTSVEmbed_Dataset

# CC 3M
CC_3M_TSV = '/comp_robot/cv_public_dataset/ConceptualCaptions/train.tsv'
# CC_3M_RESIZE_TSV = '/shared_space/caohe/data/CC3M/train_resize.tsv'
# CC_3M_RESIZE_TRAIN_SPLIT = '/shared_space/caohe/data/CC3M/train.lineidx'
# CC_3M_RESIZE_TEST_SPLIT = '/shared_space/caohe/data/CC3M/test.lineidx'
CC_3M_RESIZE_TSV = '/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv'
CC_3M_RESIZE_TRAIN_SPLIT = '/comp_robot/jiananw/ConceptualCaptions/train.lineidx'
CC_3M_RESIZE_TEST_SPLIT = '/comp_robot/jiananw/ConceptualCaptions/test.lineidx'

# CC 12M 
CC_12M_TSV = '/comp_robot/cv_public_dataset/CC12M/tsv_all/cc12m.tsv'
CC_12M_RESIZE_TSV = '/comp_robot/cv_public_dataset/CC12M/tsv_all_resize/cc12m.tsv'

# YFCC 
YFCC_TSV = '/comp_robot/cv_public_dataset/YFCC-100M/train/yfcc100m_subset_data_from_clip_data.tsv'
YFCC_RESIZE_TSV = '/comp_robot/cv_public_dataset/YFCC-100M/train/yfcc100m_subset_data_from_clip_data_resize.tsv'

# CLICKTURE
CLICKTURE_TSV = '/comp_robot/cv_public_dataset/clickture_tsv/Clickture-Full.tsv'

# Imagenet 22K 
IMAGENET_22K_TSV = '/comp_robot/cv_public_dataset/imagenet22k_tsv_new/train_all.tsv'

# Imagenet 1K 
IMAGENET_1K_TRAIN_TSV = '/comp_robot/cv_public_dataset/imagenet1k_tsv/train.tsv'
IMAGENET_1K_VAL_TSV = '/comp_robot/cv_public_dataset/imagenet1k_tsv/val.tsv'

# CoCo 2017 
COCO_2017_TRAIN_TSV = '/comp_robot/jiananw/CoCo2017/coco_train.tsv'
COCO_2017_VAL_TSV = '/comp_robot/jiananw/CoCo2017/coco_val.tsv'

class Dataset(enum.Enum):
    """
    Available datasets.
    """

    CC_3M_ORIGINAL = CC_3M_TSV  # BAD, missing lineidx file now.
    CC_3M = CC_3M_RESIZE_TSV

    CC_12M_ORIGINAL = CC_12M_TSV
    CC_12M = CC_12M_RESIZE_TSV

    YFCC_ORIGINAL = YFCC_TSV
    YFCC = YFCC_RESIZE_TSV

    CLICKTURE = CLICKTURE_TSV  # WARNINIG: small resolution 100x.

    IN_22K = IMAGENET_22K_TSV

    IN_1K = IMAGENET_1K_TRAIN_TSV
    IN_1K_VAL = IMAGENET_1K_VAL_TSV

    # self-processed
    COCO_17 = COCO_2017_TRAIN_TSV
    COCO_17_VAL = COCO_2017_VAL_TSV
    
dataset_map = {
    'CC_3M': Dataset.CC_3M,
    'CC_12M': Dataset.CC_12M,
}


class Split(enum.Enum):
    ALL = enum.auto()
    TRAIN = enum.auto()
    TEST = enum.auto()
    
class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


# Legacy datasets written in BGR
BGRDatasets = [
    Dataset.CC_3M_ORIGINAL, Dataset.CC_3M_ORIGINAL, Dataset.CC_3M,
    Dataset.YFCC_ORIGINAL, Dataset.YFCC,
    Dataset.CLICKTURE, Dataset.IN_22K,
    Dataset.IN_1K, Dataset.IN_1K_VAL,
    Dataset.COCO_17, Dataset.COCO_17_VAL,
    Dataset.CC_12M_ORIGINAL, Dataset.CC_12M
]

def get_dataset(    
        dataset:str,
        split:str,
        image_transforms=[dict(type='ToTensor')],
        tokenizer=None,
    ):
    assert dataset in dataset_map, f"Unknown dataset: {dataset}"
    dataset = dataset_map[dataset]
    lineidx_file = None  # By default the same prefix as dataset tsv file.
    if split != 'all':
        assert dataset == Dataset.CC_3M and split in ["train", "val"]
        if split == 'train':
            lineidx_file = CC_3M_RESIZE_TRAIN_SPLIT
        elif split == 'val':
            lineidx_file = CC_3M_RESIZE_TEST_SPLIT
    
    tsv_file = dataset.value
    map_color = dataset in BGRDatasets
        
    dataset = MMTSVDataset(
        tsv_file=tsv_file, 
        lineidx_file=lineidx_file, 
        map_color=map_color, 
        image_transforms=image_transforms,
        tokenizer=tokenizer,
        key='text', )
    return dataset

def get_embed_dataset(
        dataset:str,
        split='all',
        image_transforms=[dict(type='ToTensor')],
        tokenizer=None,
        clip=None, 
        clip_process=None
    ):
    assert dataset in dataset_map, f"Unknown dataset: {dataset}"
    dataset = dataset_map[dataset]
    lineidx_file = None  # By default the same prefix as dataset tsv file.
    if split != 'all':
        assert dataset == Dataset.CC_3M and split in ["train", "val"]
        if split == 'train':
            lineidx_file = CC_3M_RESIZE_TRAIN_SPLIT
        elif split == 'val':
            lineidx_file = CC_3M_RESIZE_TEST_SPLIT
    
    tsv_file = dataset.value
    map_color = dataset in BGRDatasets
    dataset = MMTSVEmbed_Dataset(
        tsv_file=tsv_file, lineidx_file=lineidx_file, 
        map_color=map_color, image_transforms=image_transforms,
        key='text', clip=clip, clip_process=clip_process,
        tokenizer=tokenizer,
    )
    return dataset