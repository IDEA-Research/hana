'''
Description: 
version: 
Author: ciao
Date: 2022-05-24 17:20:11
LastEditTime: 2022-05-26 13:26:49
'''
deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    # "optimizer": {
    #     "type": "Adam",
    #     "params": {
    #         "lr": 1.2e-4,
    #         "betas": [0.9, 0.999],
    #         "eps": 1e-8,
    #         "weight_decay": 1e-9,
    #     },
    # },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "last_batch_iteration": -1,
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 10000,
        },
    },
    "zero_optimization": {
        "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
        "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            }, # off-load cpu
        "contiguous_gradients": True,  # Reduce gradient fragmentation.
        "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
        "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
        "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "steps_per_print": 100,
    "train_micro_batch_size_per_gpu": 8,
    
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": False,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
}