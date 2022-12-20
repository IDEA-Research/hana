# Experiment config example
In summary, the pre-defined config settings containing `model`, `diffusion`, `train`, `validate`, `optimizer`, `logger`, `checkpoint` and `data`. 

Here we list an explicit explanation for text2img `64*64` experiment config as follows:

## model
```yaml
model:
  image_size: 64                    # model input image size
  num_channels: 192                 # model base channel 
  num_res_blocks: 3                 # num of residual blocks per layer
  channel_mult: "1,2,3,4"           # channel multiplier
  attention_resolutions: "32,16,8"  # resolution level where insert attention layer
  num_heads: 1                      # num of attention heads in each attention layer
  num_head_channels: 64             # if specified, ignore num_heads and instead use a fixed channel width per attention head                
  num_heads_upsample: -1            
  use_scale_shift_norm: True        # use a FiLM-like conditioning mechanism
  dropout: 0.1                      
  use_pretrained_text_encoder: True # use T5 text encodings
  use_clip_emb: False               # use CLIP embeddings
  text_ctx: 256                     # token length(L) of T5 text encodings
  xf_width: 1024                    # token dimension(D) of T5 text encodings
  xf_final_ln: True                 # apply LayerNorm to T5 text encodings
  resblock_updown: True             # use residual blocks for up/downsampling
  use_fp16: True                    # use half precision mode for training
  inpaint: False                    # inpaint model
  super_res: False                  # super-resolution model
  learn_sigma: True                 # diffusion related: learn variance, if True double the out_channels
  noise_cond_augment: False         # use noise condition augment
```

# diffusion
```yaml
diffusion:
  steps: 1000                       # diffusion total time (T)
  learn_sigma: True                 # learn variance
  sigma_small: False                # model variance type, if True fixed_small
  noise_schedule: "cosine"          # diffusion noise schedule
  use_kl: False                     # use kl-divergence loss
  predict_xstart: False             # target change to predict x_start
  rescale_timesteps: True           
  rescale_learned_sigmas: True
  schedule_sampler: null            # Default: UniformSampler
  timestep_respacing: ""            # use fewer timesteps over the same diffusion schedule 
  eval_timestep_respacing: "250"    # use fewer timesteps over the same diffusion schedule when sampling
```

# train
```yaml
train: 
  seed: 0                           # random seed
  ema:  
    enable: True                    # use ema
    decay: 0.9999                   # ema rate decay
    cpu: False                      # ema model on CPU
    update_after_steps: 1000        # after which steps ema start update 
    update_every_steps: 1           # ema update frequency
  max_epochs: 50                    # train max epochs
  max_steps: null                   
  gpus: 1                           # num GPUs to be allocated per node
  num_nodes: 1                      # num nodes to be allocated 
  gradient_clip_val: 1.0            # gradient clip value
  accumulate_grad_batches: 1        # gradient accumulate step 
  resume:
    ckpt_path: Null                 # if resume training, specify resume checkpoint path
```

# validate
```yaml
validate:
  every: 250                        # validate frequency
  dynamic_thresholding_percentile: Null # dynamic thresholding percentile e.g. 0.9
  guidance_scale: 3.0               # classifier-free guidance scale
  online: False                     # whether to eval online model.
```

# optimizer
Default to AdamW optimizer.
```yaml
optimizer:
  lr: 1.2e-4                        # learning rate
  min_lr: 1.0e-5                    # min learning rate for warmup
  betas: [0.9, 0.999]               # AdamW optimizer beta1, beta2
  eps: 1e-8                         # AdamW optimizer epsilon
  weight_decay: 1e-9                # weight decay
  warmup_num_steps: 1000            # warmup steps
```

# logger
```yaml
logger:                       
  enable: True                      # enable logger api, default logger is wandb
  project: "PROJECT_NAME"           # wandb project name
  name: "EXP_NAME"                  # wandb experiment name
  dir: "path/to/logger_dir"         # logger directory
  log_every_n_steps: 1              # log frequency
```

# checkpoint
```yaml
checkpoint:
  every_n_train_steps: 10000        # save checkpoint frequency
  dirpath: "path/to/ckpt_dir"       # checkpoint directory
  filename: "model-{step:02d}"      # checkpoint filename
  save_last: True                   # save last checkpoint
  save_top_k: -1                    # save top k checkpoint
  save_weights_only: False          # save model weights only (no optimizer states)
```

# data
```yaml
data:
  mapping_file: "path/to/mapping_file" # mapping file path (can be override by args)
  image_size: 64                       # image size, same as model image_size
  batch_size: 1                        # train batch size (can be override by args)
  val_batch_size: 1                    # validate batch size (can be override by args)
  num_workers: 4                       # num workers for dataloader
  p_flip: 0.5                          # random horizonal flip probability
  interpolation: "bilinear"            # interpolation method
```