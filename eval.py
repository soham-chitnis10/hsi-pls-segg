import torch.nn as nn
import torch
import scipy.io as sio
import utils 
import matplotlib.pyplot as plt
from model import HSI_CNN
model = HSI_CNN()
model.load_state_dict(torch.load('models/best_model.pth'))
data = sio.loadmat('data/prep_data.mat')
hycube = data['prep_sample']
hycube_y = data['prep_mask']
h = hycube.shape[0]
w = hycube.shape[1]
X,y = utils.preprocess(hycube,hycube_y,split=False,wsize=25)
pred_img = torch.zeros((h,w))
id =0
for i in range(h):
    for j in range(w):
        patch = X[id,:,:,:]
        patch = torch.unsqueeze(patch,0)
        logit = model(patch)
        y_score,y_pred = logit.max(dim=1)
        pred_img[i,j] = y_pred
        id+=1
predicted_img = pred_img.numpy()
plt.figure()
plt.imshow(predicted_img)
plt.show()
