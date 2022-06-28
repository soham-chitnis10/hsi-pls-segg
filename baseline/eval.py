import torch.nn as nn
import torch
import scipy.io as sio
import utils 
import matplotlib.pyplot as plt
from model import HSI_CNN
from dataset import HsiDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score,confusion_matrix,accuracy_score
import numpy as np
import seaborn as sns
def report(model,filename=None):
    test_dataset = HsiDataset(training=False)
    test_loader = DataLoader(test_dataset,4751)
    it = test_loader._get_iterator()
    X_test,y_test = it.next()
    logits = model(X_test)
    _,y_pred = logits.max(dim=1)
    y_test = y_test.numpy()
    y_pred = y_pred.numpy()
    cls_report = classification_report(y_test,y_pred)
    oa = accuracy_score(y_test,y_pred)
    ckappa = cohen_kappa_score(y_test,y_pred)
    confusion = confusion_matrix(y_test,y_pred)
    list_diag = np.diag(confusion)
    list_row_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(list_diag/list_row_sum)
    classes = ["Background","PS","PA6","PP","ABS"]
    sns_plot = sns.heatmap(confusion,xticklabels=classes,yticklabels=classes)
    plt.show()
    print(cls_report)
    print(f'OA: {oa}')
    print(f'Cohen Kappa Score: {ckappa}')
    print(f'\nEach class accuracy')
    
    for (a,b) in zip(classes,each_acc):
        print(f"{a}: {b}")
    if filename:
        with open(filename,'w') as f:
            print(cls_report,file=f)
            print(f'OA: {oa}',file=f)
            print(f'Cohen Kappa Score: {ckappa}',file =f)
            print(f'\nEach class accuracy',file=f)
            for (a,b) in zip(classes,each_acc):
                print(f"{a}: {b}",file=f)
            f.close()
def visulize(model,hycube,hycube_y):
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

def main():
    model = HSI_CNN()
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    # data = sio.loadmat('data/prep_data.mat')
    # hycube = data['prep_sample']
    # hycube_y = data['prep_mask']
    report(model,'results/report.txt')

if __name__ == "__main__":
    main()