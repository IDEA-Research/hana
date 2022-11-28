""""
    Multi-Modal Embedding Datasets available at IDEA.
    Data is served with (image_embedding, text_embedding, image, text) pairs. 

    # Dataset count statistics (may contain invalid images):
        Dataset.CC_3M/Split.ALL: 2,621,526
        Dataset.CC_3M/Split.TRAIN: 2,359,373
        Dataset.CC_3M/Split.TEST: 262,153
"""
import enum
import math
import os
import torch
import datetime
from torchvision.transforms import ToTensor
from torch.utils.data import ConcatDataset
from .dataset.clip_t5_embedding_dataset import CLIPT5EmbeddingDataset, LAIONT5CLIPEmbeddingDataset
from .dataset.embedding_dataset import EmbeddingDataset
from .dataset.sr_dataset import SRDataset, LAIONSRDataset
from .dataset.embedding_folder_dataset import EmbeddingFolderDataset, SRFolderDataset
from .dataset.sr_dataset import SR_ImageProcessing
# for copy large datasets to local
from .utils import copy_tsv, write_flag, read_flag, new_yaml_file

__all__ = ['get_dataset', 'get_SR_dataset']

# CC 3M
class CC_3M(enum.Enum):
    TSV = '/comp_robot/cv_public_dataset/ConceptualCaptions/train_resize.tsv'
    CLIP_BIN_FILE = "/comp_robot/cv_public_dataset/clip_embedding/cc3m/embedding.bin"
    CLIP_MAP_FILE = "/comp_robot/cv_public_dataset/clip_embedding/cc3m/embedding.txt"
    T5_BIN_FILE = "/comp_robot/cv_public_dataset/t5_embedding/cc3m/embedding.bin"
    T5_MAP_FILE = "/comp_robot/cv_public_dataset/t5_embedding/cc3m/embedding.txt"

# CC 12M
class CC_12M(enum.Enum):
    TSV = '/comp_robot/cv_public_dataset/CC12M/tsv_all_resize/cc12m.tsv'
    CLIP_BIN_FILE = '/comp_robot/cv_public_dataset/clip_embedding/cc12m/embedding.bin'
    CLIP_MAP_FILE = '/comp_robot/cv_public_dataset/clip_embedding/cc12m/embedding.txt'
    T5_BIN_FILE = "/comp_robot/cv_public_dataset/t5_embedding/cc12m/embedding.bin"
    T5_MAP_FILE = "/comp_robot/cv_public_dataset/t5_embedding/cc12m/embedding.txt"

# LAION 400M (360)
class LAION_400M(enum.Enum):
    YAML_FILE_FULL = "/comp_robot/cv_public_dataset/embedding_yaml/laion400m_without_resize.yml"
    YAML_FILE_TEST = "/comp_robot/mm_generative/data_test/test_laion_400.yml"
    CLIP_BIN_FOLDER = "/comp_robot/cv_public_dataset/clip_embedding/laion400m/"
    T5_BIN_FOLDER = "/comp_robot/cv_public_dataset/t5_embedding/laion400m/"
    LOCAL_PATH = '/raid/mm_generative/laion400m'
    LOG_PATH = '/comp_robot/mm_generative/log/laion400m'

# LAION-Aesthetics V1 subset.
# Contains the 8M most aesthetic samples (score > 8)
class LAION_ART(enum.Enum):
    YAML_FILE_FULL = "/comp_robot/cv_public_dataset/embedding_yaml/laion5B_art_without_resize.yml"
    YAML_FILE_TEST = "/comp_robot/mm_generative/data_test/test_laion_art.yml"
    CLIP_BIN_FOLDER = "/comp_robot/cv_public_dataset/clip_embedding/laion5b-art/"
    T5_BIN_FOLDER = "/comp_robot/cv_public_dataset/t5_embedding/laion5b-art/"
    LOCAL_PATH = '/raid/mm_generative/laion5b_art'
    LOG_PATH = '/comp_robot/mm_generative/log/laion5b_art'

# LAION-Aesthetics V1 subset.
# Contains 120M aesthetic samples (score > 7)
class LAION_Aesthetics_V1(enum.Enum):
    YAML_FILE_FULL = "/comp_robot/cv_public_dataset/embedding_yaml/laion5B_aesthetic_without_resize.yml"
    YAML_FILE_TEST = "/comp_robot/mm_generative/data_test/test_laion_a1.yml"
    CLIP_BIN_FOLDER = "/comp_robot/cv_public_dataset/clip_embedding/laion5b-aesthetic/"
    T5_BIN_FOLDER = "/comp_robot/cv_public_dataset/t5_embedding/laion5b-aesthetic/"
    LOCAL_PATH = '/raid/mm_generative/laion5b_aesthetic_v1'
    LOG_PATH = '/comp_robot/mm_generative/log/laion5b_aesthetic_v1'


dataset_map = {
    'CC_3M': CC_3M,
    'CC_12M': CC_12M,
    'LAION_400M': LAION_400M,
    'LAION_ART': LAION_ART,
    'LAION_A1': LAION_Aesthetics_V1,
}

def get_dataset(
    dataset: str,
    image_transforms=ToTensor(),
    tokenizer=None,
    text_encoder=None,
    copy2local=False,
):
    assert dataset in dataset_map, f"Unknown dataset: {dataset}"
    dataset = dataset_map[dataset]
    
    # if use `T5` as pretrained text encoder
    if text_encoder == 'T5':
        if dataset in [LAION_ART, LAION_Aesthetics_V1, LAION_400M]:
            yaml_file = dataset.YAML_FILE_FULL.value
            clip_bin_folder = dataset.CLIP_BIN_FOLDER.value
            t5_bin_folder = dataset.T5_BIN_FOLDER.value
            target_path = dataset.LOCAL_PATH.value
            log_path = dataset.LOG_PATH.value
            ds_name = dataset.__name__
            rank = torch.distributed.get_rank()
            os.makedirs(log_path, exist_ok=True)
            with open(f"{log_path}/{rank}", "a") as log_file:
                log_file.write("%s - 1. initializing dataset.\n"%datetime.datetime.now())
            dataset = LAIONT5CLIPEmbeddingDataset(
                yaml_file=yaml_file,
                transforms=image_transforms,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )
            with open(f"{log_path}/{rank}", "a") as log_file:
                log_file.write("%s - 2. Done initializing dataset.\n"%datetime.datetime.now())
            if not copy2local:
                return dataset
            
            # copy LAION-400M to local
            # set copy target path
            clip_target_path = f"{target_path}/clip_embedding"
            t5_target_path = f"{target_path}/t5_embedding"
            dist_file = f"{log_path}/{ds_name}.file"
            os.makedirs(target_path, exist_ok=True)
            os.makedirs(clip_target_path, exist_ok=True)
            os.makedirs(t5_target_path, exist_ok=True)
            # get distributed parameters
            node_size = int(os.environ.get('SLURM_NNODES', 1))
            node_rank = int(os.environ.get('SLURM_NODEID', 0))
            ngpus_per_node = torch.cuda.device_count()
            world_size = ngpus_per_node * node_size
            rank = torch.distributed.get_rank()
            
            num_samples = math.ceil((len(dataset) - world_size) / world_size)
            total_size = num_samples * world_size
            tsv_list = dataset.get_tsv(rank, world_size, total_size)
            clip_bin_list = [os.path.join(clip_bin_folder, os.path.basename(tsv).replace(".tsv", ".bin")) for tsv in tsv_list]
            t5_bin_list = [os.path.join(t5_bin_folder, os.path.basename(tsv).replace(".tsv", ".bin")) for tsv in tsv_list]
            t5_cache_list = [os.path.join(t5_bin_folder, os.path.basename(tsv).replace(".tsv", ".cache")) for tsv in tsv_list]
            copy_tsv(rank, tsv_list, clip_bin_list, t5_bin_list, t5_cache_list, target_path, clip_target_path, t5_target_path)
            with open(f"{log_path}/{rank}", "a") as log_file:
                log_file.write("%s - 3. Done copying.\n"%datetime.datetime.now())
            yaml_file = new_yaml_file(yaml_file, rank, tsv_list, target_path, t5_target_path, clip_target_path)
            write_flag(rank, dist_file)
            print(f"rank {rank} copy done!")
            read_flag(rank, world_size, dist_file)
            print(f"rank {rank} run!")
            if rank == 0:
                os.remove(dist_file)
            return LAIONT5CLIPEmbeddingDataset(
                yaml_file=yaml_file,
                transforms=image_transforms,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

        else:
            tsv_file = dataset.TSV.value
            t5_bin_file = dataset.T5_BIN_FILE.value
            t5_map_file = dataset.T5_MAP_FILE.value
            clip_bin_file = dataset.CLIP_BIN_FILE.value
            clip_map_file = dataset.CLIP_MAP_FILE.value
            return CLIPT5EmbeddingDataset(
                tsv_file=tsv_file,
                t5_bin_file=t5_bin_file,
                t5_map_file=t5_map_file,
                clip_bin_file=clip_bin_file,
                clip_map_file=clip_map_file,
                transforms=image_transforms,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )
    # use self-trained text encoder
    else:
        if dataset == LAION_400M:
            tsv_folder = dataset.TSV_FOLDER.value
            bin_folder = dataset.CLIP_BIN_FOLDER.value
            return ConcatDataset([EmbeddingFolderDataset(
                tsv_folder=tsv_folder, bin_folder=bin_folder, partition=i,
                transforms=image_transforms,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )] for i in range(32))
        else:
            tsv_file = dataset.TSV.value
            bin_file = dataset.CLIP_BIN_FILE.value
            map_file = dataset.CLIP_MAP_FILE.value
            return EmbeddingDataset(tsv_file, bin_file, map_file,
                                    transforms=image_transforms,
                                    tokenizer=tokenizer,
                                    text_encoder=text_encoder,
                                    )

def get_SR_dataset(
    config,
    dataset: str,
    tokenizer=None,
    text_encoder=None,
    copy2local=False,
    train_HR_transforms=None,
    train_LR_transforms=None,
):
    assert dataset in dataset_map, f"Unknown dataset: {dataset}"
    dataset = dataset_map[dataset]
    
    # if use `T5` as pretrained text encoder
    if text_encoder == 'T5':
        if dataset in [LAION_ART, LAION_Aesthetics_V1, LAION_400M]:
            yaml_file = dataset.YAML_FILE_FULL.value
            clip_bin_folder = dataset.CLIP_BIN_FOLDER.value
            t5_bin_folder = dataset.T5_BIN_FOLDER.value
            target_path = dataset.LOCAL_PATH.value + 'sr'
            log_path = dataset.LOG_PATH.value + 'sr'
            ds_name = dataset.__name__
            rank = torch.distributed.get_rank()
            os.makedirs(log_path, exist_ok=True)
            with open(f"{log_path}/{rank}", "a") as log_file:
                log_file.write("%s - 1. initializing dataset.\n"%datetime.datetime.now())
            dataset = LAIONSRDataset(
                config=config,
                yaml_file=yaml_file,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                train_HR_transforms=train_HR_transforms,
                train_LR_transforms=train_LR_transforms,
            )
            with open(f"{log_path}/{rank}", "a") as log_file:
                log_file.write("%s - 2. Done initializing dataset.\n"%datetime.datetime.now())
            if not copy2local:
                return dataset
            
            # copy LAION-400M to local
            # set copy target path
            clip_target_path = f"{target_path}/clip_embedding"
            t5_target_path = f"{target_path}/t5_embedding"
            dist_file = f"{log_path}/{ds_name}.file"
            os.makedirs(target_path, exist_ok=True)
            os.makedirs(clip_target_path, exist_ok=True)
            os.makedirs(t5_target_path, exist_ok=True)
            # get distributed parameters
            node_size = int(os.environ.get('SLURM_NNODES', 1))
            node_rank = int(os.environ.get('SLURM_NODEID', 0))
            ngpus_per_node = torch.cuda.device_count()
            world_size = ngpus_per_node * node_size
            rank = torch.distributed.get_rank()
            
            num_samples = math.ceil((len(dataset) - world_size) / world_size)
            total_size = num_samples * world_size
            tsv_list = dataset.get_tsv(rank, world_size, total_size)
            clip_bin_list = [os.path.join(clip_bin_folder, os.path.basename(tsv).replace(".tsv", ".bin")) for tsv in tsv_list]
            t5_bin_list = [os.path.join(t5_bin_folder, os.path.basename(tsv).replace(".tsv", ".bin")) for tsv in tsv_list]
            t5_cache_list = [os.path.join(t5_bin_folder, os.path.basename(tsv).replace(".tsv", ".cache")) for tsv in tsv_list]
            copy_tsv(rank, tsv_list, clip_bin_list, t5_bin_list, t5_cache_list, target_path, clip_target_path, t5_target_path,
                     copy_t5=not config.sr_img_only)
            with open(f"{log_path}/{rank}", "a") as log_file:
                log_file.write("%s - 3. Done copying.\n"%datetime.datetime.now())
            yaml_file = new_yaml_file(yaml_file, rank, tsv_list, target_path, t5_target_path, clip_target_path)
            write_flag(rank, dist_file)
            print(f"rank {rank} copy done!")
            read_flag(rank, world_size, dist_file)
            print(f"rank {rank} run!")
            if rank == 0:
                os.remove(dist_file)
            return LAIONSRDataset(
                config=config,
                yaml_file=yaml_file,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                train_HR_transforms=train_HR_transforms,
                train_LR_transforms=train_LR_transforms,
            )

        else:
            tsv_file = dataset.TSV.value
            t5_bin_file = dataset.T5_BIN_FILE.value
            t5_map_file = dataset.T5_MAP_FILE.value
            clip_bin_file = dataset.CLIP_BIN_FILE.value
            clip_map_file = dataset.CLIP_MAP_FILE.value
            return SRDataset(
                config=config,
                tsv_file=tsv_file,
                t5_bin_file=t5_bin_file,
                t5_map_file=t5_map_file,
                clip_bin_file=clip_bin_file,
                clip_map_file=clip_map_file,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )
