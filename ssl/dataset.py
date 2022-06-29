import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.io as sio
from utils import preprocess
from sklearn.model_selection import train_test_split
class HsiDataset(Dataset):
    def __init__(self,aug=transforms.RandomResizedCrop(25,interpolation=transforms.InterpolationMode.BICUBIC),mode="training"):
        self.aug = aug
        self.mode = mode
        data = sio.loadmat('./data/prep_data.mat')
        self.hycube = data['prep_sample']
        self.hycube_y = data['prep_mask']
        self.X_train,self.X_test,self.X_fine,self.y_fine,self.y_test = preprocess(self.hycube,self.hycube_y)
    def __len__(self):
        if self.mode=="training":
            return len(self.X_train)
        elif self.mode == "fine":
            return len(self.X_fine)
        else:
            return len(self.y_test)
    def __getitem__(self, index):
        if self.mode=="training":
            X = self.X_train[index]
            y = torch.Tensor()
        elif self.mode=="fine":
            X = self.X_fine[index]
            y = self.y_fine[index]
        else:
            X = self.X_test[index]
            y = self.y_test[index]
        return X,y

