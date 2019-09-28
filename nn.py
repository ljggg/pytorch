# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms, datasets

import os
import sys

batch_size = 64
learning_rate = 1e-2
num_epochs = 50
use_gpu = torch.cuda.is_available()
path = 'datasets'

def data_prepare():
    
    download = True
    if os.path.exists(path):
        download = False
    
    print(download)

    
    train_dataset = datasets.FashionMNIST(
            root=path, train=True, transform=transforms.ToTensor(), download=download)
    
    test_dataset = datasets.FashionMNIST(
            root=path, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

    

class neuralnetwork(nn.Module):
    def __init__(self, input_dim, n_hidden1, n_hidden2, n_out):
        super(neuralnetwork, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(input_dim, n_hidden1),
                nn.ReLU(True)
                )
        self.layer2 = nn.Sequential(
                nn.Linear(n_hidden1, n_hidden2),
                nn.ReLU(True)
                )
        self.layer3 = nn.Sequential(
                nn.Linear(n_hidden2, n_out),
                nn.ReLU(True)
                )
        
    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        return x

def train():
    model = neuralnetwork(28*28, 300, 100, 10)
    if use_gpu:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loader, test_loader = data_prepare()
    
    for epoch in range(num_epochs):
        print('*'*10)
        print('Epoch: {}'.format(epoch))
        
        running_loss = 0.0
        running_accuracy = 0.0
        
        for i, data in enumerate(train_loader):
            img, laybel = data
            img = img.view(img.size(0), -1)
            if use_gpu:
                img.cuda()
                laybel.cuda()
                
                
                
            out = model(img)
            loss = criterion(out, laybel)
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            running_accuracy += (pred==laybel).float().mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i%300==0:
                print(f'[{epoch+1}/{num_epochs}] Loss: {running_loss/(i+1):.6f}, Acc: {running_accuracy/(i+1):.6f}')
            
        print(f'Finish {epoch+1} epoch, Loss: {running_loss/(i+1):.6f}, Acc: {running_accuracy/(i+1):.6f}')
        
        model.eval()
        eval_loss = 0.0
        eval_accuracy = 0.0
        for data in test_loader:
            img, laybel = data
            img = img.view(img.size(0), -1)
            if use_gpu:
                img = img.cuda()
                laybel = laybel.cuda()
                
            with torch.no_grad():
                out = model(img)
                loss = criterion(out, laybel)
                
            eval_loss += loss.item()
            _, pred = torch.max(out, 1)
            eval_accuracy += (pred == label).float().mean()
        print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_accuracy/len(test_loader):.6f}\n')
    torch.save(model.state_dict(), 'model/neural_network.pth')
    print('the model saved in model/neural_network.pth')
    print('down')
    
if __name__=='__main__':
    print('start train....')
    train()



            
            
            
            
            
        
        
        
        
        
        
   
        
    
