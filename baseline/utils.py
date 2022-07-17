import scipy.io as io
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from ..utils import dim_reduc,patchify,annotate
def preprocess(hycube,hycube_y,n_components=4,split=True,wsize=25):
    new_hycube = dim_reduc(hycube,n_components)
    hycube_y = annotate(hycube_y)
    X,y = patchify(new_hycube,hycube_y,wsize)
    if split:
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.80)
        X_train =np.reshape(X_train,(-1,X_train.shape[3],X_train.shape[1],X_train.shape[2]))
        X_test= np.reshape(X_test,(-1,X_test.shape[3],X_test.shape[1],X_test.shape[2]))
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        return X_train,X_test,y_train,y_test
    X = np.reshape(X,(-1,X.shape[3],X.shape[1],X.shape[2]))
    X = X.astype(np.float32)
    y = y.astype(int)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X,y

