#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/08/03 22:56
@Author: Merc2
'''
import sys
from einops.einops import rearrange

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
parser.add_argument('--gpu', required=True, type=int)
fake_args = parser.parse_args()
args = parse_config(fake_args.config)

# device = f'cuda:{fake_args.gpu}'

device = 'cuda:0'

num_cls = 1024
shape = (16,16)
args.Model.Unet.kwargs.num_classes = num_cls
args.Model.DiscreteDiffusion.kwargs.num_class = num_cls
args.Model.DiscreteDiffusion.kwargs.shape = shape

inds = torch.load('/home/hu/multidiff/segmentation_diffusion/datasets/celeb.ckpt',map_location='cpu')
# inds = torch.load('/home/hu/multidiff/segmentation_diffusion/datasets/celeb_re.ckpt',map_location='cpu')

source = inds[-3:,:].reshape(3,16,16)
target = inds[3:6,:].reshape(3,16,16)

T = 100

ckpt = "/home/hu/multidiff/lightning/wandb/run-20210711_023610-2360ccen/files/DiscreteDDPM/2360ccen/checkpoints/last.ckpt"
# ckpt = '/home/hu/multidiff/lightning/wandb/run-20210803_143637-2ssevqnr/files/DiscreteDDPM/2ssevqnr/checkpoints/last.ckpt'
# ckpt = "/home/hu/multidiff/lightning/wandb/run-20210728_231659-tsl6dv8e/files/DiscreteDDPM/tsl6dv8e/checkpoints/last.ckpt"
# ckpt = "/home/hu/multidiff/lightning/inference/celeba_re.ckpt"
state_dict = torch.load(ckpt,map_location='cpu')
model = DDDPM(args)
model.load_state_dict(state_dict['state_dict'])
model = model.to(device)
Path.mkdir(Path('/home/hu/multidiff/Test/celeb'), exist_ok=True)
index = []
with torch.no_grad():
    for lmb in tqdm(torch.arange(0,1.1,0.1)):
        img = model._denoising.interpolate_(source.to(device),target.to(device),T,lmb)
        index.append(img.cpu())
index_for_save = torch.stack(index).cpu()


torch.save(index_for_save, f'/home/hu/multidiff/Test/celeb/interpolation_{T}_.pt')
print(f'Save :interpolation.pt')



import torch
from torch.utils.data import DataLoader


from pathlib import Path
import sys


sys.path.append('/home/hu/multidiff')

from vqgan.vqgans import VQGan

from torchvision.utils import save_image, make_grid

from tqdm import tqdm

model = VQGan('/home/hu/multidiff/vqgan/official','celeb_full').cuda(0)
BATCH_SIZE=50
seq = torch.load(f'/home/hu/multidiff/Test/celeb/interpolation_{T}.pt',map_location='cpu')
seq = torch.argmax(seq,2).reshape(11,3,1,16,16)
dir_root = Path('/home/hu/multidiff/Test/celeb')/'genfig_interpolation'
Path.mkdir(dir_root,exist_ok=True)
for i in tqdm(range(0,len(seq),BATCH_SIZE)):
    inds = seq.cuda(0)
    inds = inds.reshape(-1,256)
    assert inds.shape == (33,256)
    with torch.no_grad():
        regen = model.decode(inds.cuda(0))
    regen = rearrange(regen,'(a b) c h w -> (b a) c h w', a=11, b=3)
    save_image(make_grid(regen,nrow=11),f'k{T}_.png')
    # for j in range(BATCH_SIZE if len(inds)==BATCH_SIZE else len(inds)):
    #     save_image(regen[j,:], f'{dir_root}/{i}_{j}.png')