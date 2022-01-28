#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/30 19:08
@Author: Merc2
'''
import sys

sys.path.append('/home/hu/MultiDiff/lightning')
sys.path.append('/home/hu/MultiDiff/')
sys.path.append('/home/hu/multidiff/')
sys.path.append('/home/hu/multidiff/lightning')

import torch
import torch.nn as nn

from torchvision.utils import save_image, make_grid

from torchvision.transforms import ToTensor

from model import model_entry

from vqgan.vqgans import VQGan

from tqdm import tqdm
from solver.solver import DDDPM
from pathlib import Path
from dataset.dataloader import DDPMData
import math
import argparse
import pytorch_lightning as pl
import yaml
from easydict import EasyDict

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config

parser = argparse.ArgumentParser(description='Solver')
parser.add_argument('--config', required=True, type=str)
fake_args = parser.parse_args()
args = parse_config(fake_args.config)
data = DDPMData(args.Data)
test_loader = data.test_dataloader()

num_cls = 512
shape = (16,16)
args.Model.Unet.kwargs.num_classes = num_cls
args.Model.DiscreteDiffusion.kwargs.num_class = num_cls
args.Model.DiscreteDiffusion.kwargs.shape = shape
device = 'cuda:0'
# ckpt = "/home/hu/MultiDiff/lightning/wandb/run-20210711_023610-2360ccen/files/DiscreteDDPM/2360ccen/checkpoints/last.ckpt"
# ckpt = "/home/hu/MultiDiff/lightning/state_dict/dry.ckpt"
ckpt = "/home/hu/multidiff/lightning/wandb/run-20210728_231659-tsl6dv8e/files/DiscreteDDPM/tsl6dv8e/checkpoints/last.ckpt"
state_dict = torch.load(ckpt,map_location='cpu')
model = DDDPM(args)
model.load_state_dict(state_dict['state_dict'])
model = model.to(device)
for i,bth in enumerate(test_loader):
    images = bth.to(device)
    loss = model.validation_step(images,i)
    print(f'LOSS:{loss}')

