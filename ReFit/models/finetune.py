#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/22 18:45
@Author: Merc2
'''
import sys
from typing import OrderedDict

sys.path.append('/home/hu/MultiDiff/')
from sklearn import cluster
import time
from vqgan.modules.vqvae.quantize import VectorQuantizer2
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import importlib
import torch.optim.lr_scheduler as lr_scheduler
from vqgan.modules.vqvae.resample import *
from vqgan.modules.diffusionmodules.model import Encoder, Decoder
from vqgan.modules.vqvae.quantize import GumbelQuantize
from vqgan.modules.losses.lpips import LPIPS
from vqgan.modules.discriminator.model import NLayerDiscriminator


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


class VQModel(pl.LightningModule):
    def __init__(self,
                ddconfig,
                n_embed,
                embed_dim,
                ckpt_path=None,
                max_steps = None,
                rs_weight = None,
                sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.max_steps = max_steps
        if rs_weight is not None:
            self.RS = None
        else:
            self.RS = ReservoirSampler(15000)
        self.perceptual_loss = LPIPS().eval()
        self.discriminator = NLayerDiscriminator()
        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25, sane_index_shape=sane_index_shape)
        # self.quantize = VQBottleneck(n_embed, embed_dim, 0.25, self.RS)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.index_record = []
        self.rs_weight = rs_weight
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
    
    def RSprepare(self, batch):
        with torch.no_grad():
            h = self.encoder(batch)
            h = self.quant_conv(h)
        self.RS.add(h.detach().cpu())

    # def ReEstimate(self):
    #     #
    #     # When warming up, we keep the encodings from the last epoch and
    #     # reestimate just before new epoch starts.
    #     # When quantizing, we reestimate every number of epochs or iters given.
    #     #

    #     tstart = time.time()
    #     num_clusters = self.n_embed
    #     encodings = self.RS.contents()
    #     encodings = encodings.cpu().numpy()
    #     clustered, *_ = cluster.k_means(encodings, num_clusters)
    #     self.quantize.embedding.weight.data[
    #         ...] = torch.tensor(clustered).to(self.embedding.weight.device)
    #     self.RS.reset()
    #     print(f"Done reestimating VQ embedings, took {time.time() - tstart}s")

    def check_scale(self, state_dict):
        cb_shape = state_dict['quantize.embedding.weight'].shape
        cb = state_dict['quantize.embedding.weight']
        if cb_shape != (self.n_embed, self.embed_dim) and self.rs_weight is None:
            new = torch.normal(cb.mean(0).repeat(self.n_embed,1),cb.std(0).repeat(self.n_embed,1))
            assert new.shape == (self.n_embed, self.embed_dim)
            # new[:cb_shape[0],:] = cb
            state_dict['quantize.embedding.weight'] = new
        elif self.rs_weight is not None:
            rs_weight = torch.load(self.rs_weight, map_location='cpu')
            state_dict['quantize.embedding.weight'] = torch.tensor(rs_weight)
        new_sd = OrderedDict()
        for i in state_dict.keys():
            if ('loss.perceptual_loss' in i) or ('loss.discriminator' in i) :
                new_sd[i[5:]] = state_dict[i]
            else:
                new_sd[i] = state_dict[i]
        return new_sd

    def init_from_ckpt(self, path):
        try:
            sd = torch.load(path, map_location="cpu")['state_dict']
        except:
            sd = torch.load(path, map_location="cpu")
        sd = self.check_scale(sd)
        self.load_state_dict(sd,strict=False)
        print(f"Restored from {path}")
        print(f"CodeBook Weight {sd['quantize.embedding.weight'].shape}")

    def encode(self, x):
        with torch.no_grad():
            h = self.encoder(x)
            h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h.detach())
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @torch.no_grad()
    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, info = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, info[2]

    def training_step(self, batch, batch_idx, optimizer_idx):
        # x = batch[0] #IMAGE NET  x=batch
        # x = (2 * x) - 1
        x=batch
        xrec, code_book_loss, index = self(x)
        if optimizer_idx == 0:
            rec_loss = torch.abs(x.contiguous() - xrec.contiguous())
            p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous())
            rec_loss = rec_loss +  p_loss
            logits_fake = self.discriminator(xrec.contiguous())
            g_loss = -torch.mean(logits_fake)
            aeloss = torch.mean(rec_loss) + 0.8 * g_loss +  code_book_loss.mean()
            self.log('train/aeloss',aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return aeloss
        
        if optimizer_idx == 1:
            logits_real = self.discriminator(x.contiguous().detach())
            logits_fake = self.discriminator(xrec.contiguous().detach())
            loss_real = torch.mean(F.relu(1. - logits_real))
            loss_fake = torch.mean(F.relu(1. + logits_fake))
            d_loss = 0.5 * (loss_real + loss_fake)
            self.log('train/dloss',d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return d_loss
    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["train/aeloss"])

    def validation_epoch_end(self, outputs) :
        ir = torch.cat(self.index_record)
        used_index, counts = torch.unique(ir,return_counts=True)
        usage = len(used_index[counts>=10]) / 1024.
        self.log("code usage",usage)
        self.print(f"code usage: {usage*100.}%")
        self.index_record = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # x = batch[0] #IMAGE NET  x=batch
        # x = (2 * x) - 1
        x=batch
        xrec, code_book_loss, index = self(x)
        self.index_record.append(index)
        rec_loss = torch.abs(x.contiguous() - xrec.contiguous())
        p_loss = self.perceptual_loss(x.contiguous(), xrec.contiguous())
        rec_loss = rec_loss +  p_loss
        logits_fake = self.discriminator(xrec.contiguous())
        g_loss = -torch.mean(logits_fake)
        loss = torch.mean(rec_loss) + 0.8 * g_loss +  code_book_loss.mean()
        self.log("val/loss", loss,
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        # lr = 2e-06 single GPU
        lr = 2e-06 # continue train, half loss
        # opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
        #                             list(self.decoder.parameters())+
        #                             list(self.quantize.parameters())+
        #                             list(self.quant_conv.parameters())+
        #                             list(self.post_quant_conv.parameters()),
        #                             lr=lr, betas=(0.5, 0.9))
        opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                    list(self.quantize.parameters())+
                                    list(self.quant_conv.parameters())+
                                    list(self.post_quant_conv.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                            lr=lr, betas=(0.5, 0.9))
        sch = {'scheduler':lr_scheduler.ReduceLROnPlateau(opt_ae), 'monitor':"train/aeloss"}
        return [opt_ae, opt_disc], [sch]
        # assert self.max_steps is not None, f"Must set max_steps argument"
        # scheduler = lr_scheduler.CosineAnnealingLR(opt_ae, self.max_steps)
        # return [opt_ae], [dict(scheduler=scheduler, interval='step', frequency=1)]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = batch[0].to(self.device)#IMAGE NET  x=batch
    #     x = (2 * x) - 1
    #     xrec, _ , _ = self(x)
    #     xrec = (xrec.clamp(-1., 1.) + 1) * 0.5
    #     log["inputs"] = x
    #     log["reconstructions"] = xrec
    #     return log
    
    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch.to(self.device)#IMAGE NET  x=batch
        xrec, _ , _ = self(x)
        xrec = (xrec.clamp(-1., 1.) + 1) * 0.5
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


