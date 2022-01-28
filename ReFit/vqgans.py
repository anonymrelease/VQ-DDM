#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/02 03:03
@Author: Merc2
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from pathlib import Path
import importlib

from omegaconf import OmegaConf
from einops import rearrange

import vqgan.modules as VQ

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, remap=None, sane_index_shape=False):
        super().__init__()

        self.encoder = VQ.Encoder(**ddconfig)
        self.decoder = VQ.Decoder(**ddconfig)
        
        self.quantize = VQ.VectorQuantizer2(n_embed, embed_dim, beta=0.25,)

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff        

class GumbelVQ(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, kl_weight, remap=None):
        super().__init__()
        z_channels = ddconfig["z_channels"]
        self.vocab_size = n_embed
        self.quantize = VQ.GumbelQuantize(z_channels, embed_dim,
                                                        n_embed=n_embed,
                                                        kl_weight=kl_weight, temp_init=1.0,
                                                        remap=remap)
    
    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h
    

class VQGan(nn.Module):
    def __init__(self, CACHE_PATH = None, cktp='cub_full'):
        super().__init__()

        if cktp == 'IN_full':
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'    
        elif cktp == 'IN_dry':    
            model_filename = 'new_sd.ckpt'
            config_filename = 'new_sd.yml'       
        elif cktp == 'IN_re':    
            # model_filename = 'in_1024_re.ckpt'
            model_filename = 'epoch=0-step=109299.ckpt'
            config_filename = 'vqgan.1024.config.yml'       
        elif cktp == 'IN_re_512':    
            # model_filename = 'in_1024_re.ckpt'
            model_filename = 'IN_re_512.ckpt'
            config_filename = 'celeb-re.yml'      
        elif cktp == 'celeb_full':
            model_filename = 'celeba.ckpt'
            config_filename = 'vqgan.1024.config.yml'
        elif cktp == 'celeb_dry':
            model_filename = 'celeba-dry.ckpt'
            config_filename = 'celeb-dry.yml'
        elif cktp == 'celeb_re': ##Linux Server version
            model_filename = 'celeba-re.ckpt'
            config_filename = 'celeb-re.yml'
        elif cktp == 'celeb_re_1024': ##Linux Server version
            model_filename = 'celeb_re_1024.ckpt'
            config_filename = 'vqgan.1024.config.yml'
        
        elif cktp == 'celeb_resample': ##SOTA version
            model_filename = 'celeba-resample.ckpt'
            config_filename = 'celeb-resample.yml'
        
        elif cktp == 'LSUN': ##SOTA version
            model_filename = 'lsun.ckpt'
            config_filename = 'vqgan.1024.config.yml'

        self.flag = cktp
        config_path = str(Path(CACHE_PATH) / config_filename)
        model_path = str(Path(CACHE_PATH) / model_filename)

        self.ckpt_path = model_path
        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')

        if 'state_dict' in state.keys():
            state = state['state_dict']

        model.load_state_dict(state, strict = False)

        self.model = model

        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(math.log(f)/math.log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

        print(f"Loaded VQGAN from {model_path} and {config_path}")
    
    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)


    @torch.no_grad()
    def get_vectors(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        h = self.model.encoder(img)
        h = self.model.quant_conv(h)
        assert h.size() == (b,256,16,16)
        return h


    @torch.no_grad()
    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(math.sqrt(n)))
        img = self.model.decode(z)
        if self.flag == 'cub_full':
            return img
        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        self.model(img)

        #raise NotImplemented


class VQENC(nn.Module):
    def __init__(self, n_embed=1024, embed_dim=256, remap=None, sane_index_shape=False):
        super().__init__()
        root_path = '/home/hu/MultiDiff/vqgan/official'
        config_filename = 'vqgan.1024.config.yml'      
        config_path = str(Path(root_path) / config_filename)
        model_path = '/home/hu/MultiDiff/taming-transformers/logs/2021-08-11T03-52-41_custom_vqgan/checkpoints/last.ckpt'
    
        config = OmegaConf.load(config_path)['model']

        self.encoder = VQ.Encoder(**config['params']['ddconfig'])
        
        self.quantize = VQ.VectorQuantizer2(n_embed, embed_dim, beta=0.25,
                                    sane_index_shape=False)

        self.quant_conv = torch.nn.Conv2d(config['params']['ddconfig']["z_channels"], embed_dim, 1)

        state = torch.load(model_path,map_location='cpu')
        if 'state_dict' in state.keys():
            state = state['state_dict']

        self.load_state_dict(state, strict = False)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        h = self.encoder(img)
        h = self.quant_conv(h)
        _, _, [_, _, indices] = self.quantize(h)

        return rearrange(indices, '(b n) -> b n', b = b)
