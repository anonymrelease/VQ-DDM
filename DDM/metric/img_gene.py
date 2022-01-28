#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/30 05:10
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/13 18:49
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/13 18:49
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/13 18:49
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/13 18:49
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/13 18:49
@Author: Merc2
'''
import torch
from torch.utils.data import DataLoader


from pathlib import Path
import sys


sys.path.append('/home/hu/multidiff')

from vqgan.vqgans import VQGan

from torchvision.utils import save_image, make_grid

from tqdm import tqdm
import pytorch_fid

model = VQGan('/home/hu/multidiff/vqgan/official','IN_re').cuda(0)
BATCH_SIZE=10
# seq1 = torch.load('/home/hu/multidiff/Test/celeb/1250_ids_re_1.pt', map_location='cpu')
# seq2 = torch.load('/home/hu/multidiff/Test/celeb/1250_ids_re_2.pt', map_location='cpu')
# seq3 = torch.load('/home/hu/multidiff/Test/celeb/1250_ids_re_3.pt', map_location='cpu')
# seq4 = torch.load('/home/hu/multidiff/Test/celeb/1250_ids_re_4.pt', map_location='cpu')
# seq5 = torch.load('/home/hu/multidiff/Test/celeb/1250_ids_re_5.pt', map_location='cpu')
# seq6 = torch.load('/home/hu/multidiff/Test/celeb/1250_ids_re_6.pt', map_location='cpu')
# seq = torch.cat((seq1,seq2,seq3,seq4,seq5,seq6),dim=0)
seq = torch.load('/home/hu/multidiff/Test/IN/100.pt',map_location='cpu')
# seq = torch.load('/home/hu/MultiDiff/segmentation_diffusion/datasets/celeb.ckpt', map_location='cpu')
dir_root = Path('/home/hu/multidiff/Test/IN')/'genfig_interpolation'
# dir_root = Path.cwd()/'GenFigs'/'CUB_1024_20000'
Path.mkdir(dir_root,exist_ok=True)
for i in tqdm(range(0,len(seq),BATCH_SIZE)):
    inds = seq[i:i+BATCH_SIZE]
    if len(inds) < BATCH_SIZE:
        inds = inds.reshape(len(inds),-1)
    else:
        inds = inds.reshape(BATCH_SIZE,-1)
    with torch.no_grad():
        regen = model.decode(inds.cuda(0))
    for j in range(BATCH_SIZE if len(inds)==BATCH_SIZE else len(inds)):
        save_image(regen[j,:], f'{dir_root}/{i}_{j}.png')

# python -m pytorch_fid /home/hu/MultiDiff/Test/celeb/genfig_re_newckpt /home/hu/database/CelebA/data256x256/ --device cuda:0 --batch-size 300
