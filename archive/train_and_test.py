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


# define validate function
def validate(model, train_loader, test_loader):
    model.eval()
    # accuracy on training data and test data
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        correct = 0
        total = 0
        total_predicted = []
        total_labels = []
        
        with torch.no_grad(): # do not want gradients here, as we will not want to update parameters
            for imgs, labels in loader:
                # move data to GPU if available
                imgs = imgs.to(device=device)  
                labels = labels.to(device=device)
                total_labels.append(labels)
                
                # feed input to models
                outputs = model(imgs)  
                
                # gives the index of the highest value as output
                _, predicted = torch.max(outputs, dim=1)  
                total_predicted.append(predicted)
                
                # counts the number of example, total is increased by the batch size
                total += labels.shape[0]  
                
                # comparing the predicted class that had the maximum probability and the ground-truth labels,
                # we first get a Boolean array. Taking the sum gives the number of items in the batch where 
                # the prediction and ground truth agree
                correct += int((predicted == labels).sum()) 
                
        total_predicted = torch.hstack(total_predicted).cpu()
        total_labels = torch.hstack(total_labels).cpu()
        
        print("Accuracy {}: {:.2f}".format(name, correct / total))  
        print(sklearn.metrics.classification_report(total_labels, total_predicted))
        print(sklearn.metrics.confusion_matrix(total_labels, total_predicted))    
    

# define train_loop function
def train_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_dataloader: # loop over batches in dataset
            # move data to GPU if available
            imgs = imgs.to(device=device)  
            labels = labels.to(device=device)
            
            outputs = model(imgs)  # feed a batch through our model
            
            loss = loss_fn(outputs, labels)  # computes the loss
            
            optimizer.zero_grad()  # getting rid of the gradients from the last round
            
            loss.backward()  # performs backward step, compute the gradients of all parameters
            
            optimizer.step()  # updates the model
            
            loss_train += loss.item() # sums of losses we saw over the epoch
            
        # print the average loss per batch, in epoch%10 == 0 
        if epoch == 1 or epoch % 5 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch, loss_train/len(train_loader)
            ))
    
    

if __name__ == '__main__':
    # load dataset, normarlize it
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4990, 0.4567, 0.4188], std=[0.2913, 0.2778, 0.2836]) 
    ])

    # 0 -> mask  1-> nonmask  2 -> not a person
    train_dataset = ImageFolder('./data/train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = ImageFolder('./data/test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")
    
    model = CNN().to(device=device)  # instantiates cnn model
    
    # perform training
    learning_rate = 0.001
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # use cross entropy loss function

    # call train_loop() function
    train_loop(
        n_epochs = 20,
        optimizer = optimizer,
        model = model,
        loss_fn = loss_fn,
        train_loader = train_dataloader
    )
    torch.save(model, 'my_model.pkl')
    
    # measuring accuracy
    validate(model, train_dataloader, test_dataloader)
    
    
    
    
    
    
    
    
    
    