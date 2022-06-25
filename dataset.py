import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import scipy.io as sio
from utils import preprocess
from sklearn.model_selection import train_test_split
class HsiDataset(Dataset):
    def __init__(self,transform=ToTensor(),training=True):
        self.transform = transform
        self.training = training
        data = sio.loadmat('./data/prep_data.mat')
        self.hycube = data['prep_sample']
        self.hycube_y = data['prep_mask']
        self.X_train,self.X_test,self.y_train,self.y_test = preprocess(self.hycube,self.hycube_y)
    def __len__(self):
        if self.training:
            return len(self.y_train)
        return len(self.y_test)
    def __getitem__(self, index):
        if self.training:
            X = self.X_train[index]
            y = self.y_train[index]
        else:
            X = self.X_test[index]
            y = self.y_test[index]
        return X,y

