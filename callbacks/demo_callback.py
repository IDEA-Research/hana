'''
Description: 
version: 
Author: ciao
Date: 2022-05-24 20:18:18
LastEditTime: 2022-05-27 00:50:09
'''
from typing import List, Tuple, Dict, Union, Optional
from contextlib import contextmanager
import math
import wandb
import time

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


class DemoCallback(pl.Callback):
    def __init__(self, demo_every=2000, dynamic_thresholding_percentile=None,
                 guidance_scale=3., use_clip_emb=True, eval_online=False):
        super().__init__()
        self.demo_every = demo_every
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.guidance_scale = guidance_scale
        self.use_clip_emb = use_clip_emb
        # whether to evaluate online model when ema model is present
        self.eval_online = eval_online

    @rank_zero_only
    @th.no_grad()
    def on_batch_end(self, trainer: "pl.Trainer", module) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.demo_every != 0:
            return
        start_time = time.time()
        device = module.device

        val_dataloader = trainer.datamodule.val_dataloader()
        test_dataloader = trainer.datamodule.test_dataloader()
        val_batch = next(iter(val_dataloader))
        test_batch = next(iter(test_dataloader))
        (test_t5_encodings, test_text_emb, _), meta = test_batch
        test_texts = list(meta['text'])

        if len(val_batch) == 5:
            img_emb, text_emb, img, text, text_encodings = val_batch
            img_emb, text_emb, img, text_encodings = map(
                lambda x: x.to(device).to(text_emb.dtype), [img_emb, text_emb, img, text_encodings])

            # concat eval data with test data.
            test_t5_encodings, test_text_emb = map(
                lambda x: x.to(device).to(text_emb.dtype), [test_t5_encodings, test_text_emb])
            text_encodings = th.cat([text_encodings, test_t5_encodings])
            clip_emb = th.cat([img_emb, test_text_emb])
            text = list(text)
            text.extend(test_texts)

            batch_size = text_encodings.shape[0]
            full_batch_size = batch_size * 2
            image_size = img.shape[-1]

            # Create for the classifier-free guidance (empty)
            out_dict = {}
            out_dict["text_encodings"] = th.cat(
                [text_encodings, th.zeros_like(text_encodings)], dim=0)
            if not self.use_clip_emb:
                clip_emb = th.zeros_like(clip_emb)
            out_dict["clip_emb"] = th.cat(
                [clip_emb, th.zeros_like(clip_emb)], dim=0)
            out_dict["use_clip_emb"] = self.use_clip_emb

        elif len(val_batch) == 6:
            _, text_emb, img, text, token, mask = val_batch
            text_emb, img, token, mask = map(
                lambda x: x.to(device), [text_emb, img, token, mask])
            batch_size = text_emb.shape[0]
            full_batch_size = batch_size * 2
            image_size = img.shape[-1]

            # Create for the classifier-free guidance (empty)
            out_dict = {}
            out_dict["clip_emb"] = th.cat(
                [text_emb, th.zeros_like(text_emb)], dim=0)
            out_dict["tokens"] = th.cat([token, th.zeros_like(token)], dim=0)
            out_dict["mask"] = th.cat([mask, th.zeros_like(mask)], dim=0)

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
        for model_name, unet in eval_models.items():
            def model_fn(x_t, ts, **kwargs):
                guidance_scale = self.guidance_scale
                half = x_t[: len(x_t) // 2]
                combined = th.cat([half, half], dim=0)
                model_out = unet(combined, ts, mode='eval', **kwargs)
                eps, rest = model_out[:, :3], model_out[:, 3:]
                cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = th.cat([half_eps, half_eps], dim=0)
                return th.cat([eps, rest], dim=1)

            # sample
            with eval_mode(unet):
                sample = sampler(
                    model_fn,
                    shape=(full_batch_size, 3, image_size, image_size),
                    clip_denoised=True,
                    dynamic_threshold=self.dynamic_thresholding_percentile,
                    model_kwargs=out_dict,
                    device=device,
                )[:batch_size]

            # gather
            # log evaluation on trained images and eval images.
            eval_batch_size = batch_size//2
            val_test_grid = make_grid(sample, nrow=int(eval_batch_size), padding=0).cpu()
            val_test_image = to_pil_image(val_test_grid.add(1).div(2).clamp(0, 1))  # PIL.Image

            captions = [f"{i+1}: {cap}" for i, cap in enumerate(text)]
            log_dict[f"{model_name}_val_test_{module.eval_sampler}{num_diffusion_steps}_demo"] = wandb.Image(
                val_test_image, caption="\n".join(captions))

        optimizer = trainer.optimizers[0]
        lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        optimizer_name = optimizer.optimizer.__class__.__name__
        log_dict[optimizer_name] = lr
        
        ref_grid = make_grid(img.float(), nrow=int(eval_batch_size), padding=0).cpu()
        ref_image = to_pil_image(ref_grid)  # PIL.Image
        log_dict["GT_val"] = wandb.Image(ref_image, caption="\n".join(captions[:eval_batch_size]))

        eval_time = int(time.time() - start_time)
        log_dict["eval_time"] = eval_time

        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        del out_dict


class DemoSRCallback(pl.Callback):
    def __init__(self, demo_every=2000,
                 dynamic_thresholding_percentile=None,
                 guidance_scale=3.,
                 eval_online=False,
                 lowres_sample_noise_level=None,
                 condition_on_img_only=True,
                 eval_on_test=False,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.guidance_scale = guidance_scale
        # whether to evaluate online model when ema model is present
        self.eval_online = eval_online
        self.lowres_sample_noise_level = lowres_sample_noise_level
        self.condition_on_img_only = condition_on_img_only
        self.eval_on_test = eval_on_test

    @rank_zero_only
    @th.no_grad()
    def on_batch_end(self, trainer: "pl.Trainer", module) -> None:
        if trainer.global_step == 0 or trainer.global_step % self.demo_every != 0:
            return
        start_time = time.time()
        device = module.device

        if not self.eval_on_test:
            mode = 'eval'
            val_dataloader = trainer.datamodule.val_dataloader()
            val_batch = next(iter(val_dataloader))
            if self.condition_on_img_only:
                img, low_res_img, text = val_batch
            else:
                img, low_res_img, t5_text_encodings, text, *_ = val_batch

        else:
            mode = 'test'
            test_dataloader = trainer.datamodule.test_dataloader()
            test_batch = next(iter(test_dataloader))
            test_img, test_low_res_img, (test_t5_encodings, test_text_emb, _), meta = test_batch
            test_texts = list(meta['text'])
            img = test_img
            low_res_img = test_low_res_img
            text = list(test_texts)

        diffusion = module.eval_diffusion
        sampler = diffusion.ddim_sample_loop if module.eval_sampler =='ddim' else diffusion.p_sample_loop
        num_diffusion_steps = len(diffusion.use_timesteps)

        low_res_img = low_res_img.to(device)
        batch_size = img.shape[0]
        image_size = img.shape[-1]
        aug_t = None
        # if use noise conditional augmentation
        if self.lowres_sample_noise_level is not None:
            aug_t = th.full((batch_size,), int(diffusion.num_timesteps * self.lowres_sample_noise_level),
                            device=device)
            # I prefer lr noise augmentation to be done here outside of diffusion code.
            # But it's already handled in `p_sample_loop`, drop it here for now.
            # low_res_img = diffusion.q_sample(low_res_img, aug_t)
            augmented_img = diffusion.q_sample(low_res_img.mul(2).sub(1), aug_t).add(1).div(2).clamp(0, 1)
            upsampled = F.interpolate(
                augmented_img, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False
            ).cpu()
        else:
            upsampled = F.interpolate(
                low_res_img, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False
            ).cpu()

        low_res_img = low_res_img.mul(2).sub(1)
        out_dict = dict(
            low_res=low_res_img
        )

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
        for model_name, unet in eval_models.items():
            # sample
            with eval_mode(unet):
                sample = sampler(
                    unet,
                    shape=(batch_size, 3, image_size, image_size),
                    clip_denoised=True,
                    dynamic_threshold=self.dynamic_thresholding_percentile,
                    model_kwargs=out_dict,
                    device=device,
                    aug_t=aug_t,
                )[:batch_size]

            # gather
            # log evaluation on trained images and eval images.
            eval_batch_size = batch_size
            val_test_grid = make_grid(sample, nrow=int(eval_batch_size), padding=0).cpu()
            val_test_image = to_pil_image(val_test_grid.add(1).div(2).clamp(0, 1))  # PIL.Image

            captions = [f"{i+1}: {cap}" for i, cap in enumerate(text)]
            log_dict[f"{model_name}_{mode}_{module.eval_sampler}{num_diffusion_steps}_demo"] = wandb.Image(
                val_test_image, caption="\n".join(captions))

        optimizer = trainer.optimizers[0]
        lr = trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        optimizer_name = optimizer.optimizer.__class__.__name__
        log_dict[optimizer_name] = lr

        ref_grid = make_grid(img.float(), nrow=int(eval_batch_size), padding=0).cpu()
        ref_image = to_pil_image(ref_grid)  # PIL.Image
        log_dict[f"GT_{mode}"] = wandb.Image(ref_image, caption="\n".join(captions))

        ref_grid = make_grid(upsampled.float(), nrow=int(eval_batch_size), padding=0).cpu()
        ref_image = to_pil_image(ref_grid)  # PIL.Image
        log_dict[f"Input_{mode}"] = wandb.Image(ref_image, caption="\n".join(captions))

        eval_time = int(time.time() - start_time)
        log_dict["eval_time"] = eval_time

        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        del out_dict
