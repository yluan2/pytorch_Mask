import os
import torch
import numpy as np
import torchvision
from torch.utils import data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import visdom


transform = transforms.Compose([
    transforms.Resize((224, 224)),   #缩放到224 * 224
    transforms.CenterCrop(256),   #中心剪裁后四周padding补充
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])#均值为0 方差为1 的正态分布
])

# 0 -> mask  1-> nonmask  [tensor][label]
train_dataset = ImageFolder('./data/train', transform=transform)
# train 每四个为一组
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)






