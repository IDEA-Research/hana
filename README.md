# Hana
**Hana** is an open-source library that can create realistic images and art from a description in natural language based on **diffusion models**. It mainly follows [Imagen](https://imagen.research.google/) and [DALLE-2](https://openai.com/dall-e-2/) and built upon [pytorch-lightning](https://www.pytorchlightning.ai/) framework.

<div align="center">
  <img src="./assets/text_to_image.png" width="100%"/>
</div>

The major features of **hana** can be summarized as follows:

## News
* **`16 Dec, 2022`:** **Hana** code and pretrained weights are released!


## Installation
- Clone this repo
```bash
git clone https://github.com/IDEA-Research/hana.git
cd hana
```

- Create a conda virtual environment and activate it
```bash
conda create -n hana python=3.8 -y
conda activate hana
```

- Install `CUDA==11.1` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install required packages
```bash
pip install -r requirements.txt
```

## Dataset
Prepare your **text & img** dataset before training. Because our model will condition on the pretrained language model, we need to process and store the relevant required features in an offline manner in order to facilitate the train process. 
See [DATA](dataset/README.md) for reference.

## Config System
Please refer to [CONFIG](CONFIG.md) for the details about the basic usage and settings of experiment configs.

## Training
**Train text2img 64\*64 resolution model** 
Example shown in [train_base64.sh](./train_base64.sh)
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
- `--mapping_file [Required]`: specify dataset index file path.
- `--config_path [Required]`: specify experiment config file path.
- `--train_micro_batch_size_per_gpu:int [Required]`: batch size to be processed by one GPU in one training step (without gradient accumulation).
- `--val_batch_size:int [Required]`: validation batch size.
- `--gpus:int [Required]`: the number of GPUs to be allocated per node.
- `--num_nodes:int [Required]`: the number of nodes to be allocated.
- `--fp16`: whether to use half-precision in training.
- `--wandb_debug`: whether logging to wandb server


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

## Inference
We simply provide text2img 256*256 inference jupyter notebook. Have fun with [inference](inference.ipynb).
Before inference, please download our pretrained model weights and corresponding config files following [MODEL](MODEL.md) guidance.

## Models
Here we provide our pretrained model weights and config files, please see [MODEL](MODEL.md).

## Acknowledgements
- [GLIDE](https://github.com/openai/glide-text2im)
- [Imagen-pytorch](https://github.com/cene555/Imagen-pytorch)

## Citation
if hana is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this project:
```
@misc{ideacvr2022detrex,
  author =       {hana contributors},
  title =        {hana},
  howpublished = {\url{https://github.com/IDEA-Research/hana}},
  year =         {2022}
}
```
