'''
Description: 
version: 
Author: ciao
Date: 2022-05-23 09:39:09
LastEditTime: 2022-05-25 14:41:18
'''
import os
from typing import Sequence, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.nn import Module


class EMAWeightUpdate(Callback):
    """EMA weight update
    Your model should have:
        - ``self.online_network``
        - ``self.target_network``
    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.
    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step
    Example::
        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...
        trainer = Trainer(callbacks=[EMAWeightUpdate()])
    """

    def __init__(self, 
                 tau: float = 0.9999, 
                 update_ema_after_steps: int = 1000,
                 update_every_steps: int = 10,
                 cpu = False,
                ):
        """
        Args:
            tau: EMA decay rate
        """
        super().__init__()
        self.tau = tau
        self.update_ema_after_steps = update_ema_after_steps
        self.update_every_steps = update_every_steps
        self.cpu = cpu

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.model
        target_params = pl_module.ema_params

        if trainer.global_step >=self.update_ema_after_steps:
            if trainer.global_step == self.update_ema_after_steps:
                # copy online model weights.
                self.update_weights(online_net, target_params, decay=0.)
            elif trainer.global_step % self.update_every_steps == 0:
                self.update_weights(online_net, target_params, decay=self.tau)

    def update_weights(
        self, online_net: Union[Module, Tensor], ema_weights: Union[Module, Tensor], decay: float,
    ) -> None:
        # apply MA weight update
        with torch.no_grad():
            for name, param in online_net.named_parameters():
                if self.cpu:
                    ema_weights[name].mul_(decay).add_(param.cpu(), alpha=1-decay)
                else:
                    ema_weights[name] = ema_weights[name].to(param.device)
                    ema_weights[name].mul_(decay).add_(param, alpha=1-decay)