#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/08/22 03:42
@Author: Merc2
'''
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from vqgan.data.base import CelebA
from vqgan.vqgans import VQGan


device = 'cuda:0'
model = VQGan(r'D:\VQ-DDM\vqgan\official','LSUN').to(device)
root = r"D:\Database\lsun\church_outdoor_train"
train = CelebA(root)
DL = DataLoader(train,20,False)

for i,img in enumerate(tqdm(DL)):
    img = img.to(device)
    with torch.no_grad():
        recons = model.model.forward(img)[0].detach()
    recons = (recons.clamp(-1., 1.) + 1) * 0.5
    for j in range(len(recons)):
        save_image(recons[j,:],fr"D:\VQ-DDM\lsun_recon\{i}_{j}.png")
    if i == 2500:
        break