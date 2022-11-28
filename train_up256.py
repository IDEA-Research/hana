import argparse
import copy
import os
import copy
from omegaconf import OmegaConf
from typing import List, Tuple, Dict, Union, Optional, Any
from resize_right import resize

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

from config.deepspeed import deepspeed_config
from callbacks import EMAWeightUpdate, DemoSRCallback
from datamodule import get_SRdatamodule

from decoder.resample import LossAwareSampler, UniformSampler
from decoder.model_creation import create_model_and_diffusion as create_model_and_diffusion_dalle2
from decoder.model_creation import create_gaussian_diffusion
from decoder.text2im_model import Text2ImUNet
from decoder.respace import SpacedDiffusion
from utils.tokenizer import get_encoder

# experiment setting
os.environ["WANDB_SILENT"] = "true"


class Up256Decoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_cfg = config.MODEL
        self.diff_cfg = config.DIFFUSION
        self.train_cfg = config.TRAIN
        self.p2_loss = self.train_cfg.p2_loss
        self.condition_on_img_only = config.DATA.sr_img_only

        eval_timestep_respacing = self.diff_cfg['eval_timestep_respacing']
        schedule_sampler = self.diff_cfg.schedule_sampler
        del self.diff_cfg['eval_timestep_respacing']
        del self.diff_cfg["schedule_sampler"]
        model_and_diffusion_settings = OmegaConf.merge(
            self.model_cfg, self.diff_cfg)

        self.model, self.diffusion = create_model_and_diffusion_dalle2(
            **model_and_diffusion_settings,
        )
        self.eval_diff_cfg = copy.deepcopy(self.diff_cfg)
        self.eval_diff_cfg['timestep_respacing'] = eval_timestep_respacing
        if eval_timestep_respacing.startswith("ddim"):
            self.eval_sampler='ddim'
        else:
            self.eval_sampler='ddpm'
        self.eval_diff_cfg['steps'] = self.eval_diff_cfg['diffusion_steps']
        del self.eval_diff_cfg['diffusion_steps']
        self.eval_diffusion = create_gaussian_diffusion(**self.eval_diff_cfg)
        assert isinstance(self.model, Text2ImUNet)
        assert isinstance(self.diffusion, SpacedDiffusion)

        if schedule_sampler == 'LossAwareSampler':
            self.schedule_sampler = LossAwareSampler()
        else:
            self.schedule_sampler = UniformSampler(self.diffusion)

        # whether use noise conditional augmentation (suggested in Imagen and CDM)
        self.noise_cond_augment = self.model_cfg.noise_cond_augment

        self.tokenizer = get_encoder(text_ctx=self.model_cfg.text_ctx)

        if self.train_cfg.ema:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
            self.ema_params = copy.deepcopy(self.ema_model.state_dict())  # fp32
            # Manually restore `ema_params` from saved `ema_model` since it won't be checkpointed.
            # Note that restored `ema_params` would be upcasted from fp16 to fp32. 
            if self.train_cfg.trainer.ckpt_path is not None:
                self.restore_ema_params(f"{self.train_cfg.trainer.ckpt_path}/checkpoint/mp_rank_00_model_states.pt")
        else:
            self.ema_model = None

        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(
            params=self.model.parameters(),
            lr=self.train_cfg.lr,
            betas=self.train_cfg.betas,
            eps=self.train_cfg.eps,
            weight_decay=self.train_cfg.weight_decay,
        )
        return optimizer

    def training_step(self, batch):
        """
        Args:
            batch: 
                img: Tensor (B, C, H, W)
                low_res_img: Tensor (B, C, H', W')
        """
        if self.condition_on_img_only:
            img, low_res_img, text = batch
        else:
            assert(len(batch) == 6)  # only consider T5 pre-encodings now.
            img, low_res_img, t5_text_encodings, text, *_ = batch

        img = img.mul(2).sub(1)
        low_res_img = low_res_img.mul(2).sub(1)

        cond = dict(
            low_res=low_res_img,
        )

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
        
        # log loss terms
        self.log("train/loss", loss, on_step=True, rank_zero_only=True)
        if "mse" in losses:
            self.log("train/mse", losses["mse"].mean(), on_step=True, rank_zero_only=True)
        if "vb" in losses:
            self.log("train/vb", losses["vb"].mean(), on_step=True, rank_zero_only=True)
        return loss

    def validation_step(self, *args, **kwargs):
        pass

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
    arg_parser.add_argument("--config_path", type=str,
                            default="config/upsample256.yaml")
    arg_parser.add_argument(
        "--train_micro_batch_size_per_gpu", type=int, default=1)
    arg_parser.add_argument("--val_batch_size", type=int, default=0)
    arg_parser.add_argument("--gpus", type=int, default=1)
    arg_parser.add_argument("--num_nodes", type=int, default=1)
    arg_parser.add_argument("--fp16", action="store_true")
    arg_parser.add_argument("--offload", action="store_true")
    args = arg_parser.parse_args()

    # set up config
    config = OmegaConf.load(args.config_path)

    # change deepspeed config according to args & config
    deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_micro_batch_size_per_gpu
    if "scheduler" in deepspeed_config:
        deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = config.TRAIN.lr

    if args.fp16:
        config.MODEL.use_fp16 = True
    else:
        config.MODEL.use_fp16 = False
        del deepspeed_config["fp16"]

    if not args.offload:
        del deepspeed_config["zero_optimization"]["offload_optimizer"]

    ds_strategy = DeepSpeedStrategy(config=deepspeed_config)

    # set up data
    data_cfg = config.DATA
    data_cfg.batch_size = args.train_micro_batch_size_per_gpu
    if args.val_batch_size > 0:
        data_cfg.val_batch_size = args.val_batch_size
    dm = get_SRdatamodule(data_cfg)

    # set up decoder module
    decoder = Up256Decoder(config)

    # set up checkpoint callback
    ckpt_dir = config.TRAIN.checkpoint.dirpath
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        **config.TRAIN.checkpoint,
    )

    # demo callback
    if config.MODEL.noise_cond_augment:
        assert config.TRAIN.demo.lowres_sample_noise_level is not None, 'if noise_cond_aug enabled, need specific lowres_sample_noise_level in sampling'

    demo_callback = DemoSRCallback(
        demo_every=config.TRAIN.demo.every,
        dynamic_thresholding_percentile=config.TRAIN.demo.dynamic_thresholding_percentile,
        lowres_sample_noise_level=config.TRAIN.demo.lowres_sample_noise_level,
        condition_on_img_only=config.DATA.sr_img_only,
        eval_on_test=config.DATA.eval_test,
        eval_online=config.TRAIN.demo.online,
    )

    # ema callback
    callbacks = []
    if config.TRAIN.ema:
        ema_callback = EMAWeightUpdate(
            tau=config.TRAIN.ema_decay,
            update_ema_after_steps=config.TRAIN.update_ema_after_steps,
            update_every_steps=config.TRAIN.update_every_steps,
            cpu=config.TRAIN.cpu
        )
        callbacks.append(ema_callback)
    callbacks.extend([demo_callback, checkpoint_callback, ModelSummary(max_depth=1)])
    
    # set up wandb logger
    logger_dir = "./log/upsample256"
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        **{**config.TRAIN.wandb, 'save_dir': logger_dir, },
    )
    
    # init trainer
    pl.seed_everything(config.TRAIN.seed)
    trainer_cfg = config.TRAIN.trainer
    trainer = pl.Trainer(
        precision=16 if args.fp16 else 32,
        max_epochs=trainer_cfg.max_epochs,
        logger=wandb_logger,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        callbacks=callbacks,
        strategy=ds_strategy,
        replace_sampler_ddp=False,
    )
    trainer.logger.experiment.save(args.config_path)
    trainer.fit(decoder, datamodule=dm, ckpt_path=config.TRAIN.trainer.ckpt_path)
