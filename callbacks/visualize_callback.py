# coding=utf-8
# Copyright (c) 2022 IDEA. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wandb
import time

import torch as th
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from callbacks.utils import eval_mode

class VisualizeCallBack(pl.Callback):
    def __init__(self, demo_every=2000, dynamic_thresholding_percentile=None,
                 guidance_scale=3., eval_online=False):
        super().__init__()
        self.demo_every = demo_every
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.guidance_scale = guidance_scale
        # whether to evaluate online model when ema model is present
        self.eval_online = eval_online

    @rank_zero_only
    @th.no_grad()
    def on_batch_end(self, trainer: "pl.Trainer", module) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.demo_every != 0:
            return
        start_time = time.time()

        val_dataloader = trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_dataloader))
        
        val_img = val_batch['img']
        val_text = val_batch['caption']
        val_img_emb = val_batch.get('img_clip_embedding', None)
        val_text_emb = val_batch.get('text_clip_embedding', None)
        val_t5_encodings = val_batch.get('t5_embedding', None)
        
        device = module.device
        dtype = th.float16 if trainer.precision == 16 else th.float32
        batch_size = val_img.shape[0]
        full_batch_size = batch_size * 2
        image_size = val_img.shape[-1]
        text = list(val_text)
        
        out_dict = {}
        val_img = val_img.to(device).to(dtype)
        if val_img_emb is not None and val_text_emb is not None:
            val_text_emb = val_text_emb.to(device).to(dtype)
            out_dict['clip_emb'] = th.cat([val_text_emb, th.zeros_like(val_text_emb)], dim=0)
        else:
            out_dict['clip_emb'] = None
        
        if val_t5_encodings is not None:
            val_t5_encodings = val_t5_encodings.to(device).to(dtype)
            out_dict['text_encodings'] = th.cat([val_t5_encodings, th.zeros_like(val_t5_encodings)], dim=0)
        else:
            out_dict['text_encodings'] = None

        eval_models = {}
        ema_model = module.ema_model
        if ema_model is None:
            eval_models['online'] = module.model
        else:
            if self.eval_online:
                eval_models['online'] = module.model
            ema_model.load_state_dict(module.ema_params)
            eval_models['ema'] = ema_model

        log_dict = {}
        diffusion = module.eval_diffusion
        sampler = diffusion.ddim_sample_loop if module.eval_sampler =='ddim' else diffusion.p_sample_loop
        num_diffusion_steps = len(diffusion.use_timesteps)
        for name, model in eval_models.items():
            def model_fn(x_t, ts, **kwargs):
                guidance_scale = self.guidance_scale
                half = x_t[: len(x_t) // 2]
                combined = th.cat([half, half], dim=0)
                model_out = model(combined, ts, mode='eval', **kwargs)
                eps, rest = model_out[:, :3], model_out[:, 3:]
                cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = th.cat([half_eps, half_eps], dim=0)
                return th.cat([eps, rest], dim=1)

            # sample
            with eval_mode(model):
                sample = sampler(
                    model_fn,
                    shape=(full_batch_size, 3, image_size, image_size),
                    clip_denoised=True,
                    dynamic_threshold=self.dynamic_thresholding_percentile,
                    model_kwargs=out_dict,
                    device=device,
                )[:batch_size]

            # log sample
            sample_grid = make_grid(sample, nrow=int(batch_size//2), padding=0).cpu()
            sample_image = to_pil_image(sample_grid.add(1).div(2).clamp(0, 1))
            captions = [f"{i+1}: {cap}" for i, cap in enumerate(text)]
            log_dict[f"{name}_sample_{module.eval_sampler}{num_diffusion_steps}"] = wandb.Image(
                sample_image, caption="\n".join(captions))

        # log lr
        optimizer = trainer.optimizers[0]
        lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        optimizer_name = optimizer.optimizer.__class__.__name__
        log_dict[optimizer_name] = lr
        
        # log reference image
        ref_grid = make_grid(val_img.float(), nrow=int(batch_size//2), padding=0).cpu()
        ref_image = to_pil_image(ref_grid)
        log_dict["GT_val"] = wandb.Image(ref_image, caption="\n".join(captions))

        # log time
        eval_time = int(time.time() - start_time)
        log_dict["eval_time"] = eval_time

        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        del out_dict