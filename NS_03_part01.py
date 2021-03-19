from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from torchvision.datasets import ImageFolder


def data_loader():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256)   #中心剪裁后四周padding补充 (后续可以padding)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4990, 0.4567, 0.4188], std=[0.2913, 0.2778, 0.2836])
    ])
    trainset = ImageFolder('./data/train', transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                               shuffle=True, num_workers=2)
    testset = ImageFolder('./data/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                              shuffle=False, num_workers=2)

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # in=3x256x256; out=32x256x256
            nn.Tanh(),
            nn.MaxPool2d(2),  # out=64x128x128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # in=32x128x128, out=16x128x128
            nn.Tanh(),
            nn.MaxPool2d(2),  # out=16x64x64
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # in=16x64x64, out=8x64x64
            nn.Tanh(),
            nn.MaxPool2d(2)  # out=8x32x32
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 32 * 32, 32 * 32),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(32 * 32, 1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 128),
            nn.Tanh(),
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

def train_loop(num_epochs):
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            print(loss_list)
            # Backprop and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            print(correct / total)
            acc_list.append(correct / total)
            print(i)
            if (i + 1) % 30 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                        (correct / total) * 100))


if __name__ == '__main__':

    train_loader, _ = data_loader()
    _, test_loader = data_loader()
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 4
    train_loop(num_epochs)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'
              .format((correct / total) * 100))

