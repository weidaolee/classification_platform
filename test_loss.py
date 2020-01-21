import os

import yaml
import tqdm
import torch
import torch.nn as nn
import time
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable


import torchvision
import torchvision.transforms as transforms

from dataset import NIHDataset
from nih_loss import NIHLoss



os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with open('./config/defualt.cfg', 'r') as f:
    cfg = yaml.load(f)

batch_size = 8

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
])

train_dataset = NIHDataset(cfg, transforms=train_transforms, mode='train')

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=24)

criterion = NIHLoss()
iter_train_loader = iter(train_loader)
for i in range(train_dataset.n_images):
    name, images, target = next(iter_train_loader)
    target = Variable(target).float().cuda()
    break


pred = target
loss = criterion(pred, target)

