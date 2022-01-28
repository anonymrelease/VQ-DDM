#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/10 12:48
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/10 12:48
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/10 12:48
@Author: Merc2
'''
import sys
from pytorch_lightning.accelerators import accelerator

sys.path.append('/home/hu/multidiff/lightning')

import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from solver.solver import DDDPM
from utils import parse_config
from dataset.dataloader import DDPMData
import time

def main():
    parser = argparse.ArgumentParser(description='Solver')
    parser.add_argument('--config', required=True, type=str)
    fake_args = parser.parse_args()
    args = parse_config(fake_args.config)

    a = time.time()
    data = DDPMData(args.Data)
    data.train_dataloader()
    data.test_dataloader()
    print(f'Take {time.time()-a}s load data')

    num_cls = data.n_classes
    shape = data.data_shapes
    args.Model.Unet.kwargs.num_classes = num_cls
    args.Model.DiscreteDiffusion.kwargs.num_classes = num_cls
    args.Model.DiscreteDiffusion.kwargs.shape = shape
    model = DDDPM(args)

    # ckpt = "/home/hu/multidiff/lightning/wandb/run-20210802_195228-34p8k3m0/files/DiscreteDDPM/34p8k3m0/checkpoints/last.ckpt"
    # state_dict = torch.load(ckpt,map_location='cpu')
    # model.load_state_dict(state_dict['state_dict'])
    # print(f'finish load from {ckpt}')

    wandb_logger = WandbLogger('LSUN-limit',project='DiscreteDDPM',)

    callbacks = []
    # callbacks.append(ModelCheckpoint(save_top_k=2, mode='min', monitor='val/loss'))
    callbacks.append(ModelCheckpoint(save_last=True,every_n_train_steps=1200))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(callbacks=callbacks,max_steps=args.max_steps, 
                    # resume_from_checkpoint='/home/hu/multidiff/lightning/wandb/run-20210731_143039-37bqm4gr/files/DiscreteDDPM/37bqm4gr/checkpoints/last.ckpt',
                    accelerator='ddp', gpus=[2,3,4,5,6,7],
                    # gpus=[0],
                    logger = wandb_logger, check_val_every_n_epoch=100,
                    num_sanity_val_steps=0)
    trainer.fit(model, data)

if __name__ == '__main__':
    main()