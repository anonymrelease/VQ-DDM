#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/23 21:39
@Author: Merc2
'''
import random
import torch
import torch.nn as nn
from sklearn import cluster
import time
import torch
import torch.nn as nn
import torch.distributions as distributions
from torch.autograd import Function
import torch.nn.functional as F
from einops import rearrange
import afkmc2


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.length_reduction = 1

    def forward(self, x, features_lens=None):
        if features_lens is not None:
            return (x, features_lens)
        else:
            return x


class ReservoirSampler(nn.Module):
    def __init__(self, num_samples=2048):
        super(ReservoirSampler, self).__init__()
        self.n = num_samples
        self.ttot = 0
        self.register_buffer('buffer', None)
        self.reset()

    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    #     buffer_key = prefix + 'buffer'
    #     if buffer_key in state_dict:
    #         self.buffer = state_dict[buffer_key]
    #     return super(ReservoirSampler, self
    #                  )._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset(self):
        self.i = 0
        self.buffer = None

    def add(self, samples):
        self.ttot -= time.time()
        samples = samples.detach()
        samples = rearrange(samples, 'b c h w -> (b h w) c')
        if self.buffer is None:
            self.buffer = torch.empty(
                self.n, samples.size()[-1], device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            samples = samples[slots:]
            buffer[self.i: self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)
            if not len(samples):
                print(f"Res size {self.i}")
                self.ttot += time.time()
                return

        for s in samples:
            # warning, includes right end too.
            idx = random.randint(0, self.i)
            self.i += 1
            if idx < len(buffer):
                buffer[idx] = s
        self.ttot += time.time()

    def contents(self):
        return self.buffer[:self.i]


class VQBottleneck(nn.Module):
    def __init__(self, n_e, e_dim,
                beta=0.25,
                reestimation_reservoir=None
                ):
        super(VQBottleneck, self).__init__()

        self.embedding = nn.Embedding(n_e, e_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.num_tokens = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.reestimation_reservoir = reestimation_reservoir


    def reestimate(self):
        #
        # When warming up, we keep the encodings from the last epoch and
        # reestimate just before new epoch starts.
        # When quantizing, we reestimate every number of epochs or iters given.
        #

        tstart = time.time()
        encodings = self.reestimation_reservoir.contents()
        # encodings = rearrange(encodings,'b c h w -> (b h w) c')
        encodings = encodings.cpu().numpy()
        seeds = afkmc2.afkmc2(encodings, self.num_tokens)
        clustered, *_ = cluster.k_means(encodings, self.num_tokens, init=seeds)
        self.embedding.weight.data[
            ...] = torch.tensor(clustered).to(self.embedding.weight.device)
        self.reestimation_reservoir.reset()
        print(f"Done reestimating VQ embedings, took {time.time() - tstart}s")

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        if self.reestimation_reservoir is not None:
            self.reestimate()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None


        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                torch.mean((z_q - z.detach()) ** 2)


        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()



        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
