#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/05 16:47
@Author: Merc2
'''


from .ddpm import DiscreteDiffusion
from .ddpm2 import DiscreteDiffusionX
from .ddpm_noise import DiscreteDiffusionNoise
from .unet import Unet
from .unet_noise import UnetNoise
from .unet2 import UnetX

def model_entry(config):
    return globals()[config['type']](**config['kwargs'])