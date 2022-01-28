#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/10 11:24
@Author: Merc2
'''

import math
import random
import yaml
import os
import logging
import torch
from collections import defaultdict
import numpy as np
from easydict import EasyDict
import einops as eio
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import PIL
from pathlib import Path

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def loglik_nats(model, x):
    """Compute the log-likelihood in nats."""
    return - model.log_prob(x).mean()


def loglik_bpd(model, x):
    """Compute the log-likelihood in bits per dim."""
    return - model.log_prob(x).sum() / (math.log(2) * x.shape.numel())


def elbo_nats(model, x):
    """
    Compute the ELBO in nats.
    Same as .loglik_nats(), but may improve readability.
    """
    return loglik_nats(model, x)


def elbo_bpd(model, x):
    """
    Compute the ELBO in bits per dim.
    Same as .loglik_bpd(), but may improve readability.
    """
    return loglik_bpd(model, x)
