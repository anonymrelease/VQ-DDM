#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/10 12:58
@Author: Merc2
'''

import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import numpy as np

from PIL import Image
from pathlib import Path

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl

from einops import rearrange

from torchvision import transforms as T

import pytorch_fid 


def fn_to_tensor(img):
    img = np.array(img)

    # Add channel to grayscale images.
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(img)


class BMNIST(torch.utils.data.Dataset):
    """ BINARY MNIST """
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "train.pt"
    val_file = "val.pt"
    test_file = "test.pt"

    def __init__(self, split='train', transform=None):
        self.root = Path.cwd()
        self.transform = transform
        self.split = split

        if split not in ('train', 'val', 'test'):
            raise ValueError('split should be one of {train, val, test}')

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                            ' You can use download=True to download it')

        data_file = {'train': self.training_file,
                    'val': self.val_file,
                    'test': self.test_file}[split]
        path = os.path.join(self.root, self.processed_folder, data_file)
        self.data = torch.load(path)

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = img.byte()
            # pil_img = F_transforms.to_pil_image(img)
            # pil_img = self.transform(pil_img)
            img = fn_to_tensor(img)
            img = img.long()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        processed_folder = os.path.join(self.root, self.processed_folder)
        train_path = os.path.join(processed_folder, self.training_file)
        val_path = os.path.join(processed_folder, self.val_file)
        test_path = os.path.join(processed_folder, self.test_file)
        return os.path.exists(train_path) and os.path.exists(val_path) and \
            os.path.exists(test_path)

    def _read_raw_image_file(self, path):

        raw_file = os.path.join(self.root, self.raw_folder, path)
        all_images = []
        with open(raw_file) as f:
            for line in f:
                im = [int(x) for x in line.strip().split()]
                assert len(im) == 28 ** 2
                all_images.append(im)
        return torch.from_numpy(np.array(all_images)).view(-1, 28, 28)



class CUB(torch.utils.data.Dataset):

    def __init__(self, root_dir, weight_dir=None, flag='inds'):

        with open(root_dir, 'rb') as f:
            data_ind = pickle.load(f)
        if weight_dir is not None:
            with open(weight_dir, 'rb') as f:
                self.codebook = torch.tensor(pickle.load(f))
        data = torch.tensor(data_ind['data'])
        ind = torch.tensor(data_ind['label'])-1
        self.flag = flag
        self.pair = list(zip(data, ind))

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, ind):

        flag = self.flag
        data_ind = self.pair[ind]

        data = data_ind[0].reshape(16,16)
        return data


TRANSFORM = T.Compose([T.Resize((256,256)), T.ToTensor()])
class CUBImage(torch.utils.data.Dataset):
    def __init__(self, root, transform=TRANSFORM):
        self.path = Path(root)

        self.files = sorted(self.path.glob('**/*.jpg'))
        self.T = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]
        img = Image.open(file_path).convert('RGB')
        img = self.T(img)
        return img    

class Index(torch.utils.data.Dataset):
    def __init__(self, root):
        self.path = Path(root)

        self.files = torch.load(root, map_location='cpu')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = self.files[index,:]
        img = img.reshape(1,16,16)
        return img    

class LSUNS(torch.utils.data.Dataset):
    def __init__(self, root):
        self.path = Path(root)

        self.files = torch.load(root, map_location='cpu')[:31000,:]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = self.files[index,:]
        img = img.reshape(1,16,16)
        return img  


class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, transform=TRANSFORM):
        self.path = Path(root)

        self.files = sorted(self.path.glob('**/*.jpg'))
        self.T = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_path = self.files[index]
        img = Image.open(file_path).convert('RGB')
        img = self.T(img)
        return img    

class INIndex(torch.utils.data.Dataset):
    def __init__(self, root):
        self.p = Path(root)
        self.files = sorted(self.p.glob('*.pkl'))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,index):
        f = self.files[index]
        with open(f,'rb') as F:
            a = pickle.load(F)
        assert len(a.shape) == 2
        inds = a[0]
        label = a[1]
        return inds, label

class INIndexA(torch.utils.data.Dataset):
    def __init__(self, root):
        self.p = Path(root)
        self.data = torch.load(self.p/'idxs.pt',map_location='cpu')
        self.label = torch.load(self.p/'lbs.pt',map_location='cpu')

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,index):
        inds = self.data[index,:].reshape(16,16)
        label = self.label[index]
        return inds, label

class DDPMData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.hparams = args

    @property
    def n_classes(self):
        return self.num_classes
    @property
    def data_shapes(self):
        return self.data_shape
    def _dataset(self, flag):
        if self.hparams.dataset == 'bmnist':
            self.data_shape = (28, 28)
            pil_transforms = T.ToTensor()
            self.num_classes = 2
            train = BMNIST(split='train', transform=pil_transforms)
            test = BMNIST(split='test', transform=pil_transforms)
    
        elif self.hparams.dataset == 'cubind':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/data/temp_data.ckpt'
            self.data_shape = (16, 16)
            self.num_classes = 446
            train = Index(root)
            test = Index(root)

        elif self.hparams.dataset == 'cub':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/data/data_label_pair.pkl'
            self.data_shape = (16, 16)
            self.num_classes = 1024
            train = CUB(root_dir=root)
            test = CUB(root_dir=root)
        
        elif self.hparams.dataset == 'celeba':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/celeb.ckpt'
            self.data_shape = (16, 16)
            self.num_classes = 1024
            train = Index(root)
            test = Index(root)

        elif self.hparams.dataset == 'celeba_dry':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/celeb_dry.ckpt'
            self.data_shape = (16, 16)
            self.num_classes = 323
            train = Index(root)
            test = Index(root)
        elif self.hparams.dataset == 'celeba_re':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/celeb_re.ckpt'
            self.data_shape = (16, 16)
            self.num_classes = 512
            train = Index(root)
            test = Index(root)
        elif self.hparams.dataset == 'in':
            root = '/home/hu/ImageNet'
            self.data_shape = (16, 16)
            self.num_classes = 1024
            train = INIndexA(root)
            _, test = data.random_split(train,(len(train)-1000,1000))
        elif self.hparams.dataset == 'lsun':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/lsun.ckpt'
            self.data_shape = (16, 16)
            self.num_classes = 1024
            train = Index(root)
            _, test = data.random_split(train,(len(train)-1000,1000))
        
        elif self.hparams.dataset == 'lsuns':
            root = '/home/hu/multidiff/segmentation_diffusion/datasets/lsun.ckpt'
            self.data_shape = (16, 16)
            self.num_classes = 1024
            d = LSUNS(root)
            train, test = data.random_split(d,(len(d)-1000,1000))
        
        if flag is True:
            return  train
        else:
            return test


    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()