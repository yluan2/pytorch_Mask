import os
import torch
import numpy as np
import torchvision
from torch.utils import data
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import visdom

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 缩放到224 * 224
    # transforms.CenterCrop(256),   #中心剪裁后四周padding补充 (后续可以padding)
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 均值为0 方差为1 的正态分布
])

# 0 -> mask  1-> nonmask  [tensor][label]
train_dataset = ImageFolder('./data/train', transform=transform)
# train 每四个为一组
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

"""  显示图片
to_pil_image = transforms.ToPILImage()
for image, label in train_dataloader:
    img = to_pil_image(image[0])
    img.show()
"""


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.step1 = nn.Sequential(
            # 输入深度， 输出深度，卷积核大小（应该是随意的）
            nn.Conv2d(3, 9, 3, padding=1),  # 224 + 2 - 3 + 1= 224
            # 是否将计算值覆盖 （true -> 节省运算内存 || 不写也可以）
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 224 /2 = 112
        )
        self.step2 = nn.Sequential(
            nn.Conv2d(9, 18, 5, padding=1),  # 112 + 2 - 5 + 1 = 110
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 110 / 2 = 55
        )
        self.step3 = nn.Sequential(
            nn.Conv2d(18, 36, 4, padding=1),  # 55 + 2 - 4 + 1 = 54
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 54 / 2 = 27
        )
        self.stepLinear = nn.Sequential(
            nn.Linear(27 * 27 * 36, 112),
            nn.ReLU(inplace=True),
            nn.Linear(112, 3)  # 输出分类
        )

    def forward(self, x):
        step1_out = self.step1(x)
        step2_out = self.step2(step1_out)
        step3_out = self.step3(step2_out)
        out = self.stepLinear(step3_out.view(-1, step3_out.size(0)))
        return out


model = CNN()
print(model)
