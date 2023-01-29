import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5,padding=2)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5,padding=2)
        self.fc1=nn.Linear(16*8*8,120)
        self.bn1=nn.BatchNorm1d(120)
        self.fc2=nn.Linear(120,84)
        self.bn2=nn.BatchNorm1d(84)
        self.fc3=nn.Linear(84,10)
        self.relu=nn.ReLU()
        self.softmx=nn.Softmax(dim=-1)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=torch.flatten(x,1)
        x=self.relu(self.bn1(self.fc1(x)))
        x=self.relu(self.bn2(self.fc2(x)))
        x=self.softmx(self.fc3(x))
        return x
        