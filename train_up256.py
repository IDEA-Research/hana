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

import argparse
import copy
import os
import copy
from omegaconf import OmegaConf
from typing import Dict, Any

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

from config.deepspeed import deepspeed_config
from callbacks import EMAWeightUpdate, VisualizeSRCallBack
from data.datamodule import SRDataModule
from model.resample import LossAwareSampler, UniformSampler
from model.model_creation import create_model, create_gaussian_diffusion


class Up256Decoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = create_model(**cfg.model)
        self.diffusion = create_gaussian_diffusion(
            steps=cfg.diffusion.steps,
            learn_sigma=cfg.diffusion.learn_sigma,
            sigma_small=cfg.diffusion.sigma_small,
            noise_schedule=cfg.diffusion.noise_schedule,
            use_kl=cfg.diffusion.use_kl,
            predict_xstart=cfg.diffusion.predict_xstart,
            rescale_timesteps=cfg.diffusion.rescale_timesteps,
            rescale_learned_sigmas=cfg.diffusion.rescale_learned_sigmas,
            timestep_respacing=cfg.diffusion.timestep_respacing,
        )
        self.eval_diffusion = create_gaussian_diffusion(
            steps=cfg.diffusion.steps,
            learn_sigma=cfg.diffusion.learn_sigma,
            sigma_small=cfg.diffusion.sigma_small,
            noise_schedule=cfg.diffusion.noise_schedule,
            use_kl=cfg.diffusion.use_kl,
            predict_xstart=cfg.diffusion.predict_xstart,
            rescale_timesteps=cfg.diffusion.rescale_timesteps,
            rescale_learned_sigmas=cfg.diffusion.rescale_learned_sigmas,
            timestep_respacing=cfg.diffusion.eval_timestep_respacing,
        )
        self.eval_sampler='ddim' if cfg.diffusion.eval_timestep_respacing.startswith("ddim") else 'ddpm'
        self.schedule_sampler = self.build_schedule_sampler(cfg)

        # whether use p2 loss
        self.p2_loss = cfg.train.p2_loss
        # whether use noise conditional augmentation (suggested in Imagen and CDM)
        self.noise_cond_augment = cfg.model.noise_cond_augment
        
        if cfg.train.ema.enable:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
            self.ema_params = copy.deepcopy(self.ema_model.state_dict())  # fp32
            # Manually restore `ema_params` from saved `ema_model` since it won't be checkpointed.
            # Note that restored `ema_params` would be upcasted from fp16 to fp32. 
            if cfg.train.resume.ckpt_path is not None:
                self.restore_ema_params(f"{cfg.train.resume.ckpt_path}/checkpoint/mp_rank_00_model_states.pt")
        else:
            self.ema_model = None

        self.save_hyperparameters()
        
    def build_schedule_sampler(self, cfg):
        schedule_sampler = cfg.diffusion.schedule_sampler
        if schedule_sampler == 'LossAwareSampler':
            return LossAwareSampler()
        else:
            return UniformSampler(self.diffusion)

    def configure_optimizers(self):
        cfg = self.cfg.optimizer
        optimizer = th.optim.AdamW(
            params=self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
        return optimizer

    def training_step(self, batch):
        """
        Args:
            batch (dict): 
                img: Tensor (B, C, H, W)
                low_res_img: Tensor (B, C, H', W')
                caption: List[str]
                text_clip_embedding: Tensor (B, D), optional
                img_clip_embedding: Tensor (B, D), optional
                t5_embedding: Tensor (B, L, D), optional
        """
        cond = {}
        img = batch['HR_img']
        low_res_img = batch['LR_img']
        img = img.mul(2).sub(1)
        low_res_img = low_res_img.mul(2).sub(1)
        cond['low_res'] = low_res_img
        t, weights = self.schedule_sampler.sample(img.shape[0], img.device)
        
        # sample augmentation times `s`
        if self.noise_cond_augment:
            aug_t, _ = self.schedule_sampler.sample(img.shape[0], img.device)
            aug_t = aug_t // 10  # DO not strech model capacity.
        else:
            aug_t = None

        losses = self.diffusion.training_losses(
            model=self.model,
            x_start=img,
            t=t,
            model_kwargs=cond,
            aug_t=aug_t,
            p2_loss=self.p2_loss,
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                local_ts=t, local_losses=losses["loss"].detach()
            )
        
        loss = (losses["loss"] * weights).mean()
        for loss_term in losses:
            self.log(f"train/{loss_term}", losses[loss_term].mean(), on_step=True, rank_zero_only=True)
        self.log("train/loss", loss, on_step=True, rank_zero_only=True)
        return loss

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # load latest EMA weights to EMA model to be checkpointed.
        if self.ema_model is not None:
            self.ema_model.load_state_dict(self.ema_params)

    @rank_zero_only
    def restore_ema_params(self, path):
        sd = th.load(path, map_location="cpu")
        for name in list(self.ema_params.keys()):
            self.ema_params[name] = sd['module']['module.ema_model.' + name].float()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_path", type=str, default="config/upsample256.yaml")
    arg_parser.add_argument("--mapping_file", type=str, required=True)
    arg_parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    arg_parser.add_argument("--val_batch_size", type=int, default=0)
    arg_parser.add_argument("--gpus", type=int, default=1)
    arg_parser.add_argument("--num_nodes", type=int, default=1)
    arg_parser.add_argument("--fp16", action="store_true")
    arg_parser.add_argument("--offload", action="store_true")
    arg_parser.add_argument("--wandb_debug", action="store_true")
    args = arg_parser.parse_args()

    # set up config
    cfg = OmegaConf.load(args.config_path)

    # change deepspeed config according to args & config
    deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    if "scheduler" in deepspeed_config:
        deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = cfg.optimizer.lr
        deepspeed_config["scheduler"]["params"]["warmup_min_lr"] = cfg.optimizer.min_lr
        deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = cfg.optimizer.warmup_num_steps
    if args.fp16:
        cfg.model.use_fp16 = True
    else:
        cfg.model.use_fp16 = False
        del deepspeed_config["fp16"]
    if not args.offload:
        del deepspeed_config["zero_optimization"]["offload_optimizer"]
    deepspeed_strategy = DeepSpeedStrategy(config=deepspeed_config)

    # set up data
    cfg.data.batch_size = args.train_micro_batch_size_per_gpu
    if args.val_batch_size > 0:
        cfg.data.val_batch_size = args.val_batch_size
    data = SRDataModule(
        mapping_file=cfg.data.mapping_file,
        return_clip_embedding=cfg.model.use_clip_emb,
        return_t5_embedding=cfg.model.use_pretrained_text_encoder,
        image_size=cfg.data.image_size,
        val_image_size=cfg.data.test_image_size,
        batch_size=cfg.data.batch_size,
        val_batch_size=cfg.data.val_batch_size,
        num_workers=cfg.data.num_workers,
        interpolation=cfg.data.interpolation,
        downscale_factor=cfg.data.downscale_factor,
        min_crop_factor=cfg.data.min_crop_factor,
        max_crop_factor=cfg.data.max_crop_factor,
        random_crop=cfg.data.random_crop,
        degradation=cfg.data.degradation,
        gaussian_blur=cfg.data.gaussian_blur,
        blur_prob=cfg.data.blur_prob,
        random_flip_prob=cfg.data.random_flip_prob,
    )

    # set up decoder module
    decoder = Up256Decoder(cfg)

    # build callbacks
    callbacks = []
    
    # set up checkpoint callback
    if not os.path.exists(cfg.checkpoint.dirpath):
        os.makedirs(cfg.checkpoint.dirpath, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)
    
    # set up visualize(validation) callback
    if cfg.model.noise_cond_augment:
        assert cfg.validate.lowres_sample_noise_level is not None, 'if noise_cond_aug enabled, need specific lowres_sample_noise_level in sampling'
        
    demo_callback = VisualizeSRCallBack(
        demo_every=cfg.validate.every,
        dynamic_thresholding_percentile=cfg.validate.dynamic_thresholding_percentile,
        lowres_sample_noise_level=cfg.validate.lowres_sample_noise_level,
        eval_online=cfg.validate.online,
    )

    # set up ema callback
    if cfg.train.ema.enable:
        ema_callback = EMAWeightUpdate(
            tau=cfg.train.ema.decay,
            update_ema_after_steps=cfg.train.ema.update_after_steps,
            update_every_steps=cfg.train.ema.update_every_steps,
            cpu=cfg.train.ema.cpu
        )
        callbacks.append(ema_callback)
    callbacks.extend([demo_callback, checkpoint_callback, ModelSummary(max_depth=1)])
    
    # set up wandb logger
    logger = None
    if cfg.logger.enable:
        logger_dir = cfg.logger.dir
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir, exist_ok=True)
        os.environ["WANDB_SILENT"] = "true"
        logger = WandbLogger(
            **{
                'project': cfg.logger.project,
                'name': cfg.logger.name,
                'save_dir': logger_dir, 
                'mode': 'disabled' if args.wandb_debug else 'online',
            },
        )
    
    # init trainer
    pl.seed_everything(cfg.train.seed)
    trainer = pl.Trainer(
        precision=16 if args.fp16 else 32,
        max_epochs=cfg.train.max_epochs,
        logger=logger,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        callbacks=callbacks,
        strategy=deepspeed_strategy,
    )
    trainer.logger.experiment.save(args.config_path)
    trainer.fit(decoder, datamodule=data, ckpt_path=cfg.train.resume.ckpt_path)
