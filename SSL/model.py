
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,in_channels=4) -> None:
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels*2,5)
        self.conv2 = nn.Conv2d(in_channels*2,in_channels*4,5)
        self.conv3 = nn.Conv2d(in_channels*4,in_channels*4,3)
        self.pool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

class Projection(nn.Module):
    def __init__(self,input_dim=784,output_dim=64) -> None:
        super(Projection,self).__init__()
        self.linear1 = nn.Linear(input_dim,output_dim*4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim*4,output_dim)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
class FullyConnected(nn.Module):
    def __init__(self,input_dim=784) -> None:
        super(FullyConnected,self).__init__()
        self.linear1 = nn.Linear(input_dim,256)
        self.linear2 = nn.Linear(256,64)
        self.linear3 = nn.Linear(64,16)
        self.linear4 = nn.Linear(16,5)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
        
class HSI_CNN(nn.Module):
    def __init__(self,encoder,fc) -> None:
        super(HSI_CNN,self).__init__()
        self.encoder = encoder
        self.fc = fc

    def forward(self,x):
        x = self.encoder(x)
        x = self.fc(x)
        return x