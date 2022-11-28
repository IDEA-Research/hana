import os
import time
from tqdm import tqdm
import multiprocessing
import torch.nn as nn
import torch
import glob
import clip
import argparse
from PIL import Image
import numpy as np
import math
import shutil
import yaml

from lib2to3.pgen2 import token
from transformers import T5Tokenizer, T5EncoderModel

from dataset.tsv_dataset import TSVDataset
from dataset.embedding_dataset import EmbeddingDataset
from dataset.tsv_folder_dataset import TSVFolderDataset
from dataset.embedding_folder_dataset import EmbeddingFolderDataset
from dataset.large_scale_distributed_sampler import LargeScaleDistributedSampler
from dataset.embedding_tsv_dataset_v2 import EmbeddingTSVDatasetV2
import torch.distributed as dist


def copy_tsv(rank, file_list, bin_list, target_path):
    for i, file, bin_file in zip(range(len(file_list)), file_list, bin_list):
        target_file = os.path.join(target_path, os.path.basename(file))
        target_bin_file = os.path.join(target_path, os.path.basename(bin_file))
        if not os.path.exists(target_file) or os.path.getsize(target_file) != os.path.getsize(file):
            # print(f"[{i+1}/{len(file_list)}] rank:{rank} copy {file} to {target_file}")
            shutil.copyfile(file, target_file) # copy tsv
            shutil.copyfile(os.path.splitext(file)[0] + '.lineidx', os.path.splitext(target_file)[0] + '.lineidx') #copy lineidx

            shutil.copyfile(bin_file, target_bin_file) # copy bin
            shutil.copyfile(bin_file.replace(".bin", ".txt"), target_bin_file.replace(".bin", ".txt")) # copy map.txt


def write_flag(rank, flag_file):
    with open(flag_file, 'a') as f:
        f.write("1")

def read_flag(rank, world_size, flag_file):
    while True:
        with open(flag_file, 'r') as f:
            content = f.readline()
            if len(content) == world_size:
                break
        time.sleep(5)


def new_yaml_file(yaml_file, rank, tsv_list, target_path):
    new_file = os.path.join(target_path, os.path.basename(yaml_file).replace(".yml", f"_{rank}.yml"))
    with open(yaml_file) as f, open(new_file, 'w') as ywf:
        yaml_docs = yaml.safe_load_all(f)

        for doc in yaml_docs:
            if 'dataset' in doc and doc['dataset']['data_file'] in tsv_list:
                doc['dataset']['data_file'] = os.path.join(target_path, os.path.basename(doc['dataset']['data_file']))
            yaml.safe_dump(doc, ywf)
            ywf.write("---\n")
    return new_file

def test_embedding_tsv_dataset(rank=0, world_size=1, node_rank=0, ngpus_per_node=1, dist_url=''):
    rank = ngpus_per_node * node_rank + rank
    print(node_rank, rank, dist_url, world_size)
    dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)

    target_path = "/raid/chenyihao/laion400m"
    dist_file = "/comp_robot/workspace/chenyihao/tmp.file"
    seed = 12
    # yaml_file='/comp_robot/cv_public_dataset/LAION-400M/all.yml' # full tsv
    yaml_file='/comp_robot/cv_public_dataset/laion400m/tsv_yaml/resolution_2-score_3-length_0.yml' # test
    clip_bin_dir='/comp_robot/cv_public_dataset/clip_embedding/'
    t5_bin_dir='/comp_robot/cv_public_dataset/t5_embedding/'

    dataset = EmbeddingTSVDatasetV2(yaml_file, clip_bin_dir, t5_bin_dir, seed=seed)

    num_samples = math.ceil((len(dataset) - world_size) / world_size)
    total_size = num_samples * world_size

    tsv_list = dataset.get_tsv(rank, world_size, total_size, target_path)
    bin_list = [os.path.join(bin_dir, os.path.basename(tsv).replace(".tsv", ".bin")) for tsv in tsv_list]

    os.makedirs(target_path, exist_ok=True)
    copy_tsv(rank, tsv_list, bin_list, target_path)
    yaml_file = new_yaml_file(yaml_file, rank, tsv_list, target_path)
    # print(yaml_file)
    
    write_flag(rank, dist_file)
    print(f"rank {rank} ready!")
    read_flag(rank, world_size, dist_file)
    print(f"rank {rank} run!")

    dataset = EmbeddingTSVDatasetV2(yaml_file, target_path, seed=seed)

    train_sampler = LargeScaleDistributedSampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, num_workers=12, sampler=train_sampler, drop_last=True)

    # if rank == 0:
    #     os.remove(dist_file)

    for img_embedding, text_embedding, img, text in train_loader:
        print(img_embedding.shape, text_embedding.shape, img.shape, text.shape)
        break


def get_dict_url(node_rank, hostfile):
    if node_rank == 0:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        s = socket.socket()
        s.bind(('', 0))          # Bind to a free port provided by the host.
        port = s.getsockname()[1]  # Return the port number assigned.
        dist_url = "tcp://{}:{}".format(ip, port)
        with open(hostfile, "w") as f:
            f.write(dist_url)
    else:
        import os
        import time
        while not os.path.exists(hostfile):
            time.sleep(1)
        with open(hostfile, "r") as f:
            dist_url = f.read()

    return dist_url

if __name__ == '__main__':
    node_size = int(os.environ.get('SLURM_NPROCS', 1))
    node_rank = int(os.environ.get('SLURM_PROCID', 0))
    jobid = int(os.environ.get('SLURM_JOBID', 100))

    hostfile = f'/comp_robot/workspace/chenyihao/tmp.{jobid}'
    dist_url = get_dict_url(node_rank, hostfile)
    # print(node_rank, dist_url)

    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node * node_size
    print(world_size, node_rank, ngpus_per_node, dist_url)
    torch.multiprocessing.spawn(test_embedding_tsv_dataset, nprocs=ngpus_per_node, args=(world_size, node_rank, ngpus_per_node, dist_url))
