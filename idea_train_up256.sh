#!/bin/bash
NODES=2
GPUS_PER_NODE=8
MINI_BATCH_SIZE=64

srun -N $NODES --gres gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE \
    --qos ai4cvr --cpus-per-task 30 \
    python train_up256.py \
        --config_path config/upsample256.yaml \
        --train_micro_batch_size_per_gpu $MINI_BATCH_SIZE \
        --val_batch_size 4 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NODES \
        --fp16 