#!/bin/bash
NODES=1
GPUS_PER_NODE=2
MINI_BATCH_SIZE=16
VAL_BATCH_SIZE=4
MAPPING_FILE='path/to/mapping_file'
CONFIG_PATH='path/to/config_path'

python -W ignore train_base64.py \
    --mapping_file $MAPPING_FILE \
    --config_path $CONFIG_PATH \
    --train_micro_batch_size_per_gpu $MINI_BATCH_SIZE \
    --val_batch_size $VAL_BATCH_SIZE \
    --gpus $GPUS_PER_NODE \
    --num_nodes $NODES \
    --fp16 --wandb_debug