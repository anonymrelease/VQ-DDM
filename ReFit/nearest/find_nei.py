#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/08/21 13:36
@Author: Merc2
'''
import lpips
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

trans = ToTensor()
# target = sorted(Path(r'C:\Users\merci\Desktop\new celeba').glob('*.png'))
target = [Path(r"D:\VQ-DDM\genfig_full_final\500_5.png")]
class CelebA(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.f = sorted(Path(r'D:\Database\data256x256').glob('*.jpg'))
    
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, index):
        img_b = Image.open(self.f[index])
        img_b = trans(img_b)
        img_b = 2*img_b -1
        return img_b
for ind, t in enumerate(target):
    img_a = Image.open(t)
    img_a = trans(img_a)
    img_a = 2*img_a -1
    img_a = img_a.unsqueeze(0)

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    # f = sorted(Path(r'D:\Database\data256x256').glob('*.jpg'))

    # imgs_b = [trans(Image.open(i)) for i in f]

    ds = CelebA()
    dl = DataLoader(ds,batch_size=60,shuffle=False)

    dist1 = []
    j = 0
    for img_b in tqdm(dl):
        # img_b = Image.open(i)
        # img_b = trans(img_b)
        # img_b = 2*img_b -1
        # img_b = img_b.unsqueeze(0)
        with torch.no_grad():
            lpiploss = loss_fn_vgg(img_a.cuda(),img_b.cuda())
        dist1.append(lpiploss.detach().cpu())
        j+=1

    dist1 = torch.cat(dist1)
    kk = dist1.squeeze().argsort()
    nns = [ds.f[i] for i in kk[:10]]
    nns = [Image.open(i) for i in nns]
    result=Image.new(nns[0].mode, (256*10,256), 0xffffff)
    j = 0
    for i in nns:
        result.paste(i,(256*j,0))
        j+=1
    result.save(f'nn_{ind}.png')