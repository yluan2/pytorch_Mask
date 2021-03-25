import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datetime
import sklearn.metrics

# cnn model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # in=3x256x256; out=32x256x256
            nn.ReLU(),
            nn.MaxPool2d(2), # out=32x128x128
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # in=32x128x128, out=16x128x128
            nn.ReLU(),
            nn.MaxPool2d(2), # out=16x64x64
            nn.Conv2d(16, 8, kernel_size=3, padding=1), # in=16x64x64, out=8x64x64
            nn.ReLU(),
            nn.MaxPool2d(2) # out=8x32x32
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8*32*32, 32*32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32*32, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        # conv layer
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)
        
        return x

    
def predict(model, data_loader):
    model.eval()
    total = 0
    total_predicted = []
    
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device=device)  
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)  
            total_predicted.append(predicted)
            
    total_predicted = torch.hstack(total_predicted).cpu()
    return total_predicted
    
    
if __name__ == '__main__':
    model=torch.load('my_model.pkl')
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")
    
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 缩放到224 * 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4990, 0.4567, 0.4188], std=[0.2913, 0.2778, 0.2836])])

    dataset = ImageFolder('./sample_data/test_data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(predict(model, dataloader))
    
    
    
    
    
    
    
    