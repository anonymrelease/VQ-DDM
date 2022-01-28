#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/05 14:36
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/05 14:36
@Author: Merc2
'''

import torch
import pytorch_lightning as pl

from pathlib import Path
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import math

from model import model_entry

from .utils import parse_config, elbo_bpd

import torch.optim.lr_scheduler as lr_scheduler


class DDDPM(pl.LightningModule):
    def __init__(self, parse_args):
        super().__init__()
        self.args = parse_args
        self._nn = model_entry(self.args.Model.Unet)
        self.args.Model.DiscreteDiffusion.kwargs.denoise_fn = self._nn
        self._denoising = model_entry(self.args.Model.DiscreteDiffusion)
    #     if self.args.ckpt is not None:
    #         self.init_from_ckpt()
    
    # def init_from_ckpt(self):
    #     root = self.args.ckpt
    #     state_dict = torch.load(root,map_location='cpu')['state_dict']
    #     self.load_state_dict(state_dict, strict=True)
    #     print(f"Restored from {root}")

    def forward(self, x):
        raise NotImplemented

    def training_step(self, batch, batch_idx):
        if len(batch) !=2:
            img = batch
            cond = None
        else:
            img = batch[0]
            cond = {
                    'y': batch[1]
                }
        logits = self._denoising.log_prob(img).sum() 
        loss = - logits/ (math.log(2) * img.numel())
        self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"],on_step=True, on_epoch=True, sync_dist=True)
        # self.print(f'lr:{self.trainer.optimizers[0].param_groups[0]["lr"]}')
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    # def validation_epoch_end(self, outputs):
    #     with torch.no_grad():
    #         samples_chain = self._denoising.sample_chain(25)
    #     root = Path('/home/hu/multidiff/lightning/inference')
    #     torch.save(samples_chain,f'{root}/{self.args.Data.dataset}.ckpt')

    def validation_step(self, batch, batch_idx):
        img = batch[0]
        cond = {
                'y': batch[1]
            }
        logits = self._denoising.log_prob(img).sum() 
        loss = - logits/ (math.log(2) * img.numel())
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        #print(f'validation_loss: {loss}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.000001, betas=(0.9, 0.999))
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000000,2000000,4000000], gamma=0.5)
    #     #assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
    #     #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
    #     return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]
