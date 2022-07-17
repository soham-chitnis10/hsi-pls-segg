import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.io as sio
from baseline.utils import preprocess as baseline_preprocess
from SSL.utils import preprocess as ssl_preprocess
from sklearn.model_selection import train_test_split
class HsiDataset(Dataset):
    def __init__(self,aug=transforms.RandomResizedCrop(25,interpolation=transforms.InterpolationMode.BICUBIC),mode="training",exp="baseline"):
        self.aug = aug
        self.mode = mode
        data = sio.loadmat('./data/prep_data.mat')
        self.hycube = data['prep_sample']
        self.hycube_y = data['prep_mask']
        self.exp =exp
        if self.exp == "baseline":
            self.X_train,self.X_test,self.y_train,self.y_test = baseline_preprocess(self.hycube,self.hycube_y)
        elif self.exp == "ssl":
            self.X_train,self.X_test,self.X_fine,self.y_fine,self.y_test = ssl_preprocess(self.hycube,self.hycube_y)
    def __len__(self):
        if self.mode=="training":
            return len(self.X_train)
        elif self.mode == "fine":
            return len(self.X_fine)
        else:
            return len(self.y_test)
    def __getitem__(self, index):
        if self.mode=="training":
            if self.exp =="ssl":
                X = self.X_train[index]
                y = torch.Tensor()
            else:
                X = self.X_train[index]
                y = self.y_train[index]
        elif self.mode=="fine":
            X = self.X_fine[index]
            y = self.y_fine[index]
        else:
            X = self.X_test[index]
            y = self.y_test[index]
        return X,y

