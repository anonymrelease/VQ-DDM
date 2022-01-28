#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/05 16:52
@Author: Merc2
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from inspect import isfunction
import random
from .utils import *
from tqdm import tqdm  

class DiscreteDiffusionNoise(nn.Module):
    def __init__(self, num_class, shape, denoise_fn, timesteps=4000):
        super().__init__()

        self.num_classes = num_class
        self._nn = denoise_fn
        
        self.num_timesteps = timesteps
        
        self.shape = torch.Size(shape)

        alphas = cosine_beta_schedule(timesteps)
        
        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))
    

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        # Computing alpha_t * E[ xt ] + (1 - alpha_t ) 1 / K
        # Eq 11 beta is alpha
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        # Eq 12
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )

        return log_probs

    def predict_start(self, log_x_t, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        x_t = log_onehot_to_index(log_x_t)
        x_t_fl = (x_t / (self.num_classes-1))
        n_t = self._nn(t, x_t_fl, **model_kwargs)
        x_b = F.one_hot(x_t, self.num_classes)
        n_t = n_t.squeeze()
        n_t = rearrange(n_t,'b c h w -> b c 1 h w')
        x_b = rearrange(x_b,'b 1 h w c -> b c 1 h w')

        out = n_t + x_b

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # Extention of Bayes.
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) * q(x0)/ [q(xt | x0) * q(x0)]

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)

        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        log_x_recon = self.predict_start(log_x, t=t, model_kwargs=model_kwargs)
        log_model_pred = self.q_posterior(
            log_x_start=log_x_recon, log_x_t=log_x, t=t)
        return log_model_pred
    
    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    @torch.no_grad()
    def p_sample(self, log_x, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        model_log_prob = self.p_pred(log_x=log_x, t=t, model_kwargs=model_kwargs)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def interpolate_(self, x1, x2, t = None, lam = 0.1):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        x_shape = x1.size()[1:]
        if len(x1.size()) == 3:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
        assert x1.shape == x2.shape
        log_x1_start = index_to_log_onehot(x1, self.num_classes)
        log_x2_start = index_to_log_onehot(x2, self.num_classes)
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (log_x1_start, log_x2_start))

        img = torch.log(lam*torch.exp(xt1) + (1-lam)*torch.exp(xt2))
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img


    def q_sample(self, log_x_start, t):

        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, log_x_start, log_x_t, t, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x=log_x_t, t=t, model_kwargs=model_kwargs)


        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)
        # T = 0 KLD
        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss


    def nll(self, log_x_start, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                model_kwargs=model_kwargs)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def sample_time(self, b, device, method='importance'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        b, device = x.size(0), x.device

        x_start = x

        t, pt = self.sample_time(b, device, 'importance')

        log_x_start = index_to_log_onehot(x_start, self.num_classes)

        kl = self.compute_Lt(
            log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, model_kwargs)

        Lt2 = kl.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        kl_prior = self.kl_prior(log_x_start)

        # Upweigh loss term of the kl
        vb_loss = kl / pt + kl_prior

        return -vb_loss


    def log_prob(self, x, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x, model_kwargs)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, model_kwargs)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def sample(self, num_samples, model_kwargs):
        b = num_samples
        device = self.log_alpha.device
        if model_kwargs is None:
            model_kwargs = {'y':torch.randint(0,1024,(num_samples,),device=device).long()}
        uniform_logits = torch.zeros((b, self.num_classes) + torch.Size(self.shape), device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        for i in tqdm(reversed(range(0, self.num_timesteps))):
            print(f'Sample timestep {i:4d}', end='\r')

            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, model_kwargs)
        print()
        return log_onehot_to_index(log_z)

    def sample_chain(self, num_samples, model_kwargs=None):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        ### conditional sample
        if model_kwargs is None:
            model_kwargs = {'y':torch.randint(0,1024,(num_samples,),device=device).long()}

        zs = torch.zeros((self.num_timesteps, b) + torch.Size(self.shape)).long()
        
        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, model_kwargs)

            zs[i] = log_onehot_to_index(log_z).squeeze()
        print()
        return zs