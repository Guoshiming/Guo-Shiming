# -*- coding: utf-8 -*-


import copy

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
from lightly.data import LightlyDataset

class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(2048, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

"""brain tumor"""

resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = MoCo(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = MoCoV2Transform(input_size=32)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset("Brain_Tumor_total", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss(memory_bank_size=(4096, 128))
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

epochs = 100

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    for batch in dataloader:
        x_query, x_key = batch[0]
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=momentum_val
        )
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        query = model(x_query)
        key = model.forward_momentum(x_key)
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(resnet.state_dict(), 'pretrained_Moco/brain_tumor_resnet.pth')

"""covid"""

resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = MoCo(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = MoCoV2Transform(input_size=32)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset("covid_total", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss(memory_bank_size=(4096, 128))
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

epochs = 100

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    for batch in dataloader:
        x_query, x_key = batch[0]
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=momentum_val
        )
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        query = model(x_query)
        key = model.forward_momentum(x_key)
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(resnet.state_dict(), 'pretrained_Moco/covid_resnet.pth')

"""cancer"""

resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = MoCo(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = MoCoV2Transform(input_size=32)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# or create a dataset from a folder containing images or videos:
dataset = LightlyDataset("cancer_total", transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss(memory_bank_size=(4096, 128))
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

epochs = 100

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    for batch in dataloader:
        x_query, x_key = batch[0]
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=momentum_val
        )
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        query = model(x_query)
        key = model.forward_momentum(x_key)
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(resnet.state_dict(), 'pretrained_Moco/cancer_resnet.pth')