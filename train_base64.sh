#!/bin/bash
NODES=2
GPUS_PER_NODE=8
MINI_BATCH_SIZE=128

srun -N $NODES --gres gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE \
    --qos ai4cvr-1 --cpus-per-task 30 \
    python -W ignore train_base64.py \
        --config_path config/tiny64_t5.yaml \
        --train_micro_batch_size_per_gpu $MINI_BATCH_SIZE \
        --val_batch_size 4 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NODES \
        --fp16 