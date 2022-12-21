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
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from callbacks.utils import eval_mode


class VisualizeSRCallBack(pl.Callback):
    def __init__(self, 
                 demo_every=2000,
                 dynamic_thresholding_percentile=None,
                 guidance_scale=3.,
                 eval_online=False,
                 lowres_sample_noise_level=None,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.guidance_scale = guidance_scale
        # whether to evaluate online model when ema model is present
        self.eval_online = eval_online
        self.lowres_sample_noise_level = lowres_sample_noise_level

    @rank_zero_only
    @th.no_grad()
    def on_batch_end(self, trainer: "pl.Trainer", module) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.demo_every != 0:
            return
        start_time = time.time()
        device = module.device

        mode = 'eval'
        val_dataloader = trainer.datamodule.val_dataloader()
        val_batch = next(iter(val_dataloader))
        img = val_batch['HR_img']
        low_res_img = val_batch['LR_img']
        text = val_batch['caption']
        low_res_img = low_res_img.to(device)
        batch_size = img.shape[0]
        image_size = img.shape[-1]
        aug_t = None

        diffusion = module.eval_diffusion
        sampler = diffusion.ddim_sample_loop if module.eval_sampler =='ddim' else diffusion.p_sample_loop
        num_diffusion_steps = len(diffusion.use_timesteps)
        
        # if use noise conditional augmentation
        if self.lowres_sample_noise_level is not None:
            aug_t = th.full((batch_size,), int(diffusion.num_timesteps * self.lowres_sample_noise_level),
                            device=device)
            augmented_img = diffusion.q_sample(low_res_img.mul(2).sub(1), aug_t).add(1).div(2).clamp(0, 1)
            upsampled = F.interpolate(
                augmented_img, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False
            ).cpu()
        else:
            upsampled = F.interpolate(
                low_res_img, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False
            ).cpu()

        low_res_img = low_res_img.mul(2).sub(1)
        out_dict = dict(low_res=low_res_img)

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
        for name, model in eval_models.items():
            # sample
            with eval_mode(model):
                sample = sampler(
                    model,
                    shape=(batch_size, 3, image_size, image_size),
                    clip_denoised=True,
                    dynamic_threshold=self.dynamic_thresholding_percentile,
                    model_kwargs=out_dict,
                    device=device,
                    aug_t=aug_t,
                )[:batch_size]

            # log sample
            sample_grid = make_grid(sample, nrow=int(batch_size), padding=0).cpu()
            sample_image = to_pil_image(sample_grid.add(1).div(2).clamp(0, 1))

            captions = [f"{i+1}: {cap}" for i, cap in enumerate(text)]
            log_dict[f"{name}_{mode}_{module.eval_sampler}{num_diffusion_steps}"] = wandb.Image(
                sample_image, caption="\n".join(captions))

        # log lr
        optimizer = trainer.optimizers[0]
        lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        optimizer_name = optimizer.optimizer.__class__.__name__
        log_dict[optimizer_name] = lr

        # log reference HR image
        ref_grid = make_grid(img.float(), nrow=int(batch_size), padding=0).cpu()
        ref_image = to_pil_image(ref_grid)
        log_dict[f"GT_{mode}"] = wandb.Image(ref_image, caption="\n".join(captions))

        # log upsampled image
        ref_grid = make_grid(upsampled.float(), nrow=int(batch_size), padding=0).cpu()
        ref_image = to_pil_image(ref_grid)
        log_dict[f"Input_{mode}"] = wandb.Image(ref_image, caption="\n".join(captions))

        # log time
        eval_time = int(time.time() - start_time)
        log_dict["eval_time"] = eval_time

        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        del out_dict
