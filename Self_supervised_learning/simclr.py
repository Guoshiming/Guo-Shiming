# -*- coding: utf-8 -*-


import sys
sys.path.append('/content/drive/My Drive/CS15-2 capstone')

import torch
import torch.nn as nn
import os
import matplotlib.image as mtimage
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import re, os, sys, json, random
from tqdm import tqdm
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
import shutil
import torchvision.datasets as dset
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

"""# SIMCLR

CANCER DATASET
"""


root_train='cancer_total'

########
from lightly.data import LightlyDataset
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.projection_head = SimCLRProjectionHead(2048, 512, 128)
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z



resnet = torchvision.models.resnet50()
# resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
########

transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset(root_train, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(resnet.state_dict(), 'pretrained_SIMCLR/cancer.pth')

"""COVID DATASET"""

#root path
root_train='covid_total'


########
from lightly.data import LightlyDataset
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.projection_head = SimCLRProjectionHead(2048, 512, 128)
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z



resnet = torchvision.models.resnet50()
# resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
########

transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset(root_train, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4096,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(resnet.state_dict(), 'pretrained_SIMCLR/covid.pth')

"""Brain Tumor DATASET"""

#root path
root_train='Brain_Tumor_total'


########
from lightly.data import LightlyDataset
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.projection_head = SimCLRProjectionHead(512, 512, 128)
        self.projection_head = SimCLRProjectionHead(2048, 512, 128)
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z



resnet = torchvision.models.resnet50()
# resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
########

transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset(root_train, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2048,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(resnet.state_dict(), 'pretrained_SIMCLR/Brain_tumor.pth')