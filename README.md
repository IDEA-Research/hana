
# Introduction
**xiaobai** is an open-source library that can create realistic images and art from a description in natural language based on diffusion models. It mainly follows Imagen and DALLE-2 and built upon pytorch-lightning framework.

## Updates
- (2022/12/16) Release our **xiaobai** project.

## Table of Contents
0. [Installation](#Installation)
1. [Dataset setup](#Dataset-setup)
2. [Config Introduction](#Config-Introduction)
3. [Training](#Training)
4. [Inference](#Inference)
5. [Models](#Models)

### Installation
- Clone this repo
```bash
git clone https://github.com/IDEA-Research/xiaobai.git
cd xiaobai
```

- Create a conda virtual environment and activate it
```bash
conda create -n xiaobai python=3.8 -y
conda activate xiaobai
```

- Install `CUDA==11.1` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install required packages
```bash
pip install -r requirements.txt
```

### Dataset setup
Prepare your text&img dataset before training. Because our model will condition on the pretrained language model, we need to process and store the relevant required features in an offline manner in order to facilitate the train process. 
See [data setup](dataset/README.md) for reference.

### Config Introduction
Please refer to [Config Instructions]() for the details about the basic usage and settings of experiment configs.

### Training
**Train text2img 64\*64 resolution model** 
Example shown in `train_base64.sh`
```bash
NODES=1
GPUS_PER_NODE=2
MINI_BATCH_SIZE=16
VAL_BATCH_SIZE=4
MAPPING_FILE='path/to/mapping_file'
CONFIG_PATH='path/to/config_path'

python train_base64.py --mapping_file $MAPPING_FILE --config_path $CONFIG_PATH \
    --train_micro_batch_size_per_gpu $MINI_BATCH_SIZE --val_batch_size $VAL_BATCH_SIZE \
    --gpus $GPUS_PER_NODE --num_nodes $NODES \
    --fp16 --wandb_debug
```
Args:
- `--mapping_file [Required]` specify dataset index file path.
- `--config_path [Required]` specify experiment config file path.
- `--train_micro_batch_size_per_gpu:int [Required]` batch size to be processed by one GPU in one training step (without gradient accumulation).
- `--val_batch_size:int [Required]` validation batch size.
- `--gpus:int [Required]` the number of GPUs to be allocated per node.
- `--num_nodes:int [Required]` the number of nodes to be allocated.
- `--fp16` whether to use half-precision in training.
- `--wandb_debug` whether logging to wandb server


**Resume Training**
Specify resume checkpoint path in experiment config file, like:
```yaml
...
train:
    resume:
        ckpt_path: path/to/resume_checkpoint
...
```
Then, normally run train script.

### Inference
We simply provide text2img 256*256 inference jupyter notebook. Have fun with [eval_pipeline.ipynb](eval_pipeline.ipynb)!
Before inference, please download our pretrained model weights and corresponding config files following [MODEL](MODEL.md) guidance.

### Models
Here we provide our pretrained model weights and config files, please see [MODEL.md](MODEL.md)

## Acknowledgements
- [GLIDE](https://github.com/openai/glide-text2im)
- [Imagen-pytorch](https://github.com/cene555/Imagen-pytorch)

## Citation
if xiaobai is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this project:
```
@misc{ideacvr2022detrex,
  author =       {xiaobai contributors},
  title =        {xiaobai},
  howpublished = {\url{https://github.com/IDEA-Research/xiaobai}},
  year =         {2022}
}
```
