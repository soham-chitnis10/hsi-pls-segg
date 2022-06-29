import scipy.io as io
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F

def dim_reduc(hycube,n_components):
    new_hycube =np.reshape(hycube,(-1,hycube.shape[2]))
    pca = PCA(n_components,whiten=True,random_state=0)
    new_hycube = pca.fit_transform(new_hycube)
    new_hycube =np.reshape(new_hycube,(hycube.shape[0],hycube.shape[1],n_components))
    return new_hycube

def patchify(hycube,hycube_y,wsize,padding=True):
    pad = int((wsize-1)/2)
    N = hycube.shape[0]*hycube.shape[1] 
    if padding:
        hycube = np.pad(hycube,pad_width=((pad,pad),(pad,pad),(0,0)),constant_values=0)
    new_hycube = np.zeros((N,wsize,wsize,hycube.shape[2]))
    new_hycube_y = np.zeros((N))
    id =0
    for r in range(pad,hycube.shape[0]-pad):
        for c in range(pad,hycube.shape[1]-pad):
            new_hycube[id,:,:,:] = hycube[r-pad:r+pad+1,c-pad:c+pad+1]
            new_hycube_y[id]=hycube_y[r-pad,c-pad]
            id += 1
    return new_hycube,new_hycube_y

def preprocess(hycube,hycube_y,n_components=4,split=True,wsize=25):
    new_hycube = dim_reduc(hycube,n_components)
    for i in range(203):
        for j in range(117):
            if i<100 and j>50 and hycube_y[i,j]==1:
                hycube_y[i,j]=2
            elif i>100 and j<50 and hycube_y[i,j]==1:
                hycube_y[i,j]=3
            elif i>100 and j>50 and hycube_y[i,j]==1:
                hycube_y[i,j]=4
    X,y = patchify(new_hycube,hycube_y,wsize)
    if split:
        X,X_test,y,y_test = train_test_split(X,y,train_size=0.80)
        X_train,X_fine,_,y_fine = train_test_split(X,y,test_size=0.125)
        X_train =np.reshape(X_train,(-1,X_train.shape[3],X_train.shape[1],X_train.shape[2]))
        X_test= np.reshape(X_test,(-1,X_test.shape[3],X_test.shape[1],X_test.shape[2]))
        X_fine= np.reshape(X_fine,(-1,X_fine.shape[3],X_fine.shape[1],X_fine.shape[2]))
        y_fine = y_fine.astype(int)
        y_test = y_test.astype(int)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_fine = X_fine.astype(np.float32)
        X_train = torch.from_numpy(X_train)
        y_fine = torch.from_numpy(y_fine)
        X_fine = torch.from_numpy(X_fine)
        X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)
        return X_train,X_test,X_fine,y_fine,y_test
    X = np.reshape(X,(-1,X.shape[3],X.shape[1],X.shape[2]))
    X = X.astype(np.float32)
    y = y.astype(int)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X,y

def info_nce_loss(S,temp=0.5):
    labels = torch.cat([torch.arange(S.shape[0]//2) for i in range(2)])
    labels = (labels.unsqueeze(0)==labels.unsqueeze(1)).float()
    s_norm =F.normalize(S,dim=1)
    sim = torch.matmul(s_norm,s_norm.T)
    mask = torch.eye(labels.shape[0],dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0],-1)
    sim = sim[~mask].view(sim.shape[0],-1)
    pos = sim[labels.bool()].view(labels.shape[0],-1)
    neg = sim[~labels.bool()].view(sim.shape[0],-1)
    logits = torch.cat([pos,neg],dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    logits /= temp
    return logits,labels


