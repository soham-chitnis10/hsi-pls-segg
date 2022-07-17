import scipy.io as io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import numpy as np

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

def annotate(hycube_y):
    for i in range(203):
        for j in range(117):
            if i<100 and j>50 and hycube_y[i,j]==1:
                hycube_y[i,j]=2
            elif i>100 and j<50 and hycube_y[i,j]==1:
                hycube_y[i,j]=3
            elif i>100 and j>50 and hycube_y[i,j]==1:
                hycube_y[i,j]=4
    return hycube_y

