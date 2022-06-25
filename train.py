from dataset import HsiDataset
from torch.utils.data import DataLoader
from model import HSI_CNN
from torch.optim import Adam
import torch.nn as nn
import torch
import scipy.io as sio
import utils 
import matplotlib.pyplot as plt
train_data = HsiDataset()
test_data = HsiDataset(training=False)
train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=32)
model = HSI_CNN()
optimizer = Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
best_acc = 0
acc_list = [0]
for epoch in range(50):
    model.train()
    print(f'Epoch: {epoch}')
    for i,(X,y) in enumerate(train_dataloader):
        logits = model(X)
        batch_loss = criterion(logits,y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if i%50 ==0:
            print(f'Train Loss: {batch_loss} batch_id: {i}')
    print("Starting eval")
    model.eval()
    epoch_acc = 0
    with torch.no_grad():
        for i,(X,y) in enumerate(test_dataloader):
            logits = model(X)
            batch_loss = criterion(logits,y)
            y_score,y_pred= logits.max(dim=1)
            acc = (y==y_pred).sum()/y.size(0)
            acc = acc.item()
            epoch_acc += acc
            print(f'Batch Accuracy: {acc} batch_id: {i}')
        if (epoch_acc/(i+1)) > best_acc:
            torch.save(model.state_dict(),'best_model.pth')
        print(f'Epoch Accuracy: {epoch_acc/(i+1)}')
        acc_list.append(epoch_acc/(i+1))
    
epoch_list = list(range(0,51))
plt.plot(epoch_list,acc_list)
plt.show()
