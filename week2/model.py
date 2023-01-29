import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(784,200)
        self.bn1=nn.BatchNorm1d(200)
        self.rel=nn.ReLU()
        self.fc2=nn.Linear(200,50)
        self.bn2=nn.BatchNorm1d(50)
        
        self.fc3=nn.Linear(50,10)
        self.softmx=nn.Softmax(dim=-1)
    def forward(self,x):
        x=self.rel(self.bn1(self.fc1(x)))
        x=self.rel(self.bn2(self.fc2(x)))
        x=self.softmx(self.fc3(x))
        return x

        
n_epochs = 10
batch_size_train = 150
learning_rate = 0.01
momentum = 0.9
log_interval = 20

def train(model, optimizer, criterion, train_loader, display_step=None):
    running_loss=0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        # print(loss.item(),end=" ")
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch_idx%log_interval==(log_interval-1):
            print(f'[{batch_idx + 1:5d}] loss: {running_loss :.3f}')
            running_loss=0
            
    print("finished")