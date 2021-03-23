from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
from skorch import callbacks
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score,classification_report
import sys

def data_loader():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256)   #中心剪裁后四周padding补充 (后续可以padding)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4990, 0.4567, 0.4188], std=[0.2913, 0.2778, 0.2836])
    ])
    trainset = ImageFolder('./data/train', transform=transform)
    #split train data
    # m=len(trainset)
    # train_data, val_data = random_split(trainset, [int(m-m*0.2), int(m*0.2)])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                               shuffle=True, num_workers=2)
    
    # valid_loader = torch.utils.data.DataLoader(val_data, batch_size=64)

    testset = ImageFolder('./data/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                              shuffle=False, num_workers=2)

    return trainset, train_loader, trainset, test_loader




class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # in=3x256x256; out=32x256x256
            nn.ReLU(),
            nn.MaxPool2d(2),  # out=64x128x128
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # in=32x128x128, out=16x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # out=16x64x64
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # in=16x64x64, out=8x64x64
            nn.ReLU(),
            nn.MaxPool2d(2)  # out=8x32x32
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(8 * 32 * 32, 32 * 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32 * 32, 1024),
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

    train_set,_, _,_ = data_loader()
    _,train_loader,_ , _ = data_loader()
    _,_,test_set,_ = data_loader()
    _, _, _, test_loader = data_loader()

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 4
    train_loop(num_epochs)
    model.eval()
    y_test_list = []
    y_pred_list = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred_list.append(predicted.cpu().numpy())
            y_test_list.append(labels.cpu().numpy())
        print('Test Accuracy of the model on the test images: {} %'
              .format((correct / total) * 100))
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

#Evaluate our model in Pytorch
    y_test = np.array([y for x, y in iter(test_set)])

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test_list[0],y_pred_list[0]))
    
    print("confusion matrix")
    print(confusion_matrix_df)

    print(classification_report(y_test_list[0], y_pred_list[0]))

# #Using Skorch to evaluate our CNN model
# class MaskConvNet():
#   # model code  
#   train_set,_, _,_ = data_loader()
#   _,train_loader,_ , _ = data_loader()
#   _,_,test_set,_ = data_loader()
#   _, _, _, test_loader = data_loader()
#   y_train = np.array([y for x, y in iter(train_set)])
#   # print(train_set)
#   # print(x_train)
#   # print(y_train)

#   torch.manual_seed(0)

#   net = NeuralNetClassifier(
#       CNN,
#       max_epochs=4,
#       iterator_train__num_workers=4,
#       lr=0.001,
#       batch_size=32,
#       optimizer = torch.optim.Adam,
#       criterion=nn.CrossEntropyLoss,
#     #   device=torch.device('cuda'),
#       train_split = 0  
#   )

#   net.fit(train_set, y=y_train)

#   y_pred = net.predict(test_set)
#   y_test = np.array([y for x, y in iter(test_set)])
#   # print(len(y_pred))
#   # print(len(y_test))
#   # print(accuracy_score(y_test, y_pred))
#   F1 = f1_score(y_test,y_pred,average='micro')
#   print("f1 score:", F1)
#   recall = recall_score(y_test, y_pred,average="micro")
#   print("recall: ", recall)
#   precision = precision_score(y_test,y_pred,average='micro')
#   print("precision: ", precision)
#   print("confusion matrix")
#   print(confusion_matrix(y_test, y_pred))