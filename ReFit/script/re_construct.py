#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/08/02 18:28
@Author: Merc2
'''
import sys

from torchvision.transforms import transforms

sys.path.append('/home/hu/MultiDiff')
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid
from pathlib import Path
from vqgan.vqgans import VQGan
from einops import rearrange
import time
from PIL import Image

# class CenterCropLongEdge(object):

#     def __call__(self, img):

#         return T.functional.center_crop(img, min(img.size))

#     def __repr__(self):
#         return self.__class__.__name__


# class CenterCropLongEdge(object):

#     def __call__(self, img):

#         return T.functional.center_crop(img, min(img.size))

#     def __repr__(self):
#         return self.__class__.__name__



# class LSUN(Dataset):
#     def __init__(self, root, transform):
#         self.path = Path(root)

#         self.files = sorted(self.path.glob('*.jpg'))
#         self.T = transform

#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, index):
#         file_path = self.files[index]
#         img = Image.open(file_path).convert('RGB')
#         img = self.T(img)
#         # img = img *2. -1.
#         return img    

# device = 'cuda:0'



# # IN_root = Path("/home/hu/database/ImageNet/train/")
# LSUN_root = Path("/home/hu/database/LSUN/church_outdoor_train/")
# # imgs_root = sorted(IN_root.glob("*/*.JPEG"))
# imgs_root = sorted(LSUN_root.glob("*.jpg"))
# assert len(imgs_root) == 126227
# pool_cap = 20000
# BS = 30
# device = 'cuda:0'

# # IN_r = Path("/home/hu/database/ImageNet")
# img_sample = torch.randint(0,  len(imgs_root), (1,pool_cap)) 
# idx_sample = torch.randint(0,  16*16, (1,pool_cap)) 
# trans = T.Compose([CenterCropLongEdge(), T.Resize(256),T.ToTensor()])
# # IN = ImageNet(IN_r, 'train',transform=trans)
# dataset = LSUN(LSUN_root, transform=trans)
# model = VQGan('/home/hu/MultiDiff/vqgan/official','IN_full').to(device)
# DDD = DataLoader(dataset=dataset, batch_size=BS,sampler=img_sample.squeeze())
# pool = []
# for i,batch in tqdm(enumerate(DDD)):
#     img = batch
#     vectors = model.get_vectors(img.to(device)).detach().cpu()
#     vectors = rearrange(vectors, 'b c h w -> b (h w) c')
#     vector = vectors[range(len(vectors)),idx_sample[0,i*BS:i*BS+len(vectors)],:]
#     pool.append(vector.cpu())
#     if i == 0:
#         with torch.no_grad():
#             recon,_ = model.model.forward(img.to(device))
#         save_image(make_grid(recon,nrow=3), 'wthlsun.png')

# pool = torch.cat(pool)
# torch.save(pool,'lsun_cb_.ckpt')

encodings = torch.load('lsun_cb_.ckpt').cpu().numpy()

#b,c,h,w = pool.size()
#pool = rearrange(pool,'b c h w -> (b h w) c')
#sampless = torch.randint(0,  b*h*w, (1,20000)) 
#encodings = pool[sampless.squeeze(),:].cpu().numpy()
assert encodings.shape == (20000,256)
num_clusters = 1024
import afkmc2
tstart = time.time()
seeds = afkmc2.afkmc2(encodings, num_clusters)
reclu_time = time.time()-tstart
print(f'Find Seeds Time: {reclu_time}')
torch.save(seeds,'lsun_seeds_1024.ckpt')
from sklearn.cluster import KMeans
model = KMeans(n_clusters=num_clusters, init=seeds).fit(encodings)
new_centers = model.cluster_centers_
clu_time = time.time() - tstart
print(f'Cluster Time: {clu_time}')
torch.save(new_centers,'lsun_emb_1024.ckpt')
# original_ckpt = torch.load(model.ckpt_path, map_location = 'cpu')
# if 'state_dict' in original_ckpt.keys():
#     original_ckpt = original_ckpt['state_dict']
# original_ckpt['quantize.embedding.weight'] = new_centers
# torch.save(original_ckpt,'imgnet_new.ckpt')
