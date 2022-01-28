#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/12 18:53
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

import math
import argparse

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
parser.add_argument('--gpu', required=False, type=int, default=0)
fake_args = parser.parse_args()
args = parse_config(fake_args.config)

device = f'cuda:{fake_args.gpu}'

# num_cls = 1024
shape = (16,16)
# args.Model.Unet.kwargs.num_classes = num_cls
# args.Model.DiscreteDiffusion.kwargs.num_class = num_cls
args.Model.DiscreteDiffusion.kwargs.shape = shape

# ckpt = "/home/hu/MultiDiff/lightning/wandb/run-20210711_023610-2360ccen/files/DiscreteDDPM/2360ccen/checkpoints/last.ckpt"
# ckpt = "/home/hu/MultiDiff/lightning/state_dict/dry.ckpt"
# ckpt = "/home/hu/multidiff/lightning/wandb/run-20210728_231659-tsl6dv8e/files/DiscreteDDPM/tsl6dv8e/checkpoints/last.ckpt"
ckpt = "/home/hu/multidiff/lightning/wandb/latest-run/files/DiscreteDDPM/3lg721y5/checkpoints/last.ckpt"
# ckpt = "/home/hu/multidiff/lightning/inference/celeba_re.ckpt"
state_dict = torch.load(ckpt,map_location='cpu')
model = DDDPM(args)
model.load_state_dict(state_dict['state_dict'])
model = model.to(device)
Path.mkdir(Path('/home/hu/multidiff/Test/IN'), exist_ok=True)
index = []
for _ in tqdm(range(1)):
    with torch.no_grad():
        # idx = {'y':torch.randint(0,1024,(100,),device=device).long()}
        idx = {'y':torch.ones((100,)).long().cuda()*5}
        samples_chain = model._denoising.sample(100,idx)
        assert samples_chain.shape[-2:] == (16,16)
    index.append(samples_chain.cpu())

    index_for_save = torch.cat(index).cpu()

    
    torch.save(index_for_save, f'/home/hu/multidiff/Test/IN/100.pt')
