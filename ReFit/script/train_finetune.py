#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/23 17:16
@Author: Merc2
'''
from pytorch_lightning.accelerators import accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append(r'D:\VQ-DDM')
sys.path.append(r'D:\VQ-DDMvqgan')

import afkmc2

import math
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf
from easydict import EasyDict
import yaml
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from vqgan.models.finetune import VQModel
from vqgan.data.base import VQGanData

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)


    # @rank_zero_only
    # def log_local(self, save_dir, split, images,
    #               global_step, current_epoch, batch_idx):
    #     root = os.path.join(save_dir, "images", split)
    #     for k in images:
    #         grid = torchvision.utils.make_grid(images[k], nrow=4)

    #         grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
    #         grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
    #         grid = grid.numpy()
    #         grid = (grid*255).astype(np.uint8)
    #         filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
    #             k,
    #             global_step,
    #             current_epoch,
    #             batch_idx)
    #         path = os.path.join(root, filename)
    #         os.makedirs(os.path.split(path)[0], exist_ok=True)
    #         Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # self.log_local(pl_module.logger.save_dir, split, images,
            #             pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")


class RSampler(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        if pl_module.rs_weight is not None:
            print("*********************NO Pre-RSampling!!!**************************")
        else:
            print("*********************Begin Pre-RSampling**************************")
            for img in tqdm(trainer.train_dataloader):
                pl_module.RSprepare(img.to(pl_module.device))
            # pl_module.ReEstimate()
            print("*********************Done Pre-RSampling!**************************")


def main():
    parser = argparse.ArgumentParser(description='Solver')
    parser.add_argument('--config', required=True, type=str)
    fake_args = parser.parse_args()
    args = parse_config(fake_args.config)
    data = VQGanData(args.Data)
    data.train_dataloader()
    data.test_dataloader()

    model = VQModel(**args.model)

    wandb_logger = WandbLogger(f'LSUN-RF',project='VQGan-finetune')
    image_logger = ImageLogger(batch_frequency=750, max_images=4)
    callbacks = []
    callbacks.append(ModelCheckpoint(save_top_k=2, mode='min', monitor='val/loss'))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(image_logger)
    
    # callbacks.append(RSampler())

    print(f"******************Steps:{args.model.max_steps}**********************")
    trainer = pl.Trainer(callbacks=callbacks,max_steps=args.model.max_steps, 
                    # resume_from_checkpoint='/home/hu/MultiDiff/wandb/latest-run/files/VQGan-finetune/23s5up02/checkpoints/last.ckpt',
                    logger = wandb_logger,val_check_interval=0.01, gpus=[0], 
                    # accelerator='ddp',
                    num_sanity_val_steps=0)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()