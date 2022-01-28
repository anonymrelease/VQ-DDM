#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/23 17:40
@Author: Merc2
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on: 2021/07/22 17:44
@Author: Merc2
'''
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as T


class CelebA(Dataset):
    def __init__(self, config=None):
        self.config = OmegaConf.create(config)
        root = self.config.root
        self.transform = T.ToTensor()

        self.files = sorted(Path(root).glob("*.jpg"))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_root = self.files[index]
        img = Image.open(img_root)
        img_tensor = self.transform(img)
        return img_tensor