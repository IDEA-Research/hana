#!/bin/bash
NODES=1
GPUS_PER_NODE=2
MINI_BATCH_SIZE=16

srun -N $NODES --gres gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE \
    --qos ai4cvr-1 --cpus-per-task 30 \
    python -W ignore train_base64.py \
        --mapping_file /comp_robot/mm_generative/data/cc3m/cc3m_map.json \
        --config_path config/xiaobai_tiny64.yaml \
        --train_micro_batch_size_per_gpu $MINI_BATCH_SIZE \
        --val_batch_size 4 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NODES \
        --fp16 --wandb_debug