from pickletools import optimize
from model import HSI_CNN, Encoder,Projection,FullyConnected
from dataset import HsiDataset
from torch.utils.data import DataLoader
from img_aug import augment
import torch
from utils import info_nce_loss
import torch.nn as nn
from torch.optim import Adam
import os
import matplotlib.pyplot as plt
def pre_training(encoder,projection,epochs=200):
    train_dataset = HsiDataset()
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([{"params":encoder.parameters()},{"params":projection.parameters()}],lr=0.0001)
    for epoch in range(epochs):
        epoch_loss = 0
        for i,(X,_) in enumerate(train_loader):
            X1,X2= augment(X)
            X_new = torch.concat([X1,X2])
            Z= encoder(X_new)
            S= projection(Z)
            logits,labels = info_nce_loss(S)
            loss = criterion(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        print(f'Loss: {epoch_loss/(i+1)} Epoch: {epoch+1}')
        torch.save(encoder.state_dict(),'ssl/models/pre_trained_encoder_crop_flip_color_jitter.pth')
def fine_tune(encoder,fully_connected):
    fine_dataset = HsiDataset(mode="fine")
    test_dataset = HsiDataset(mode="test")
    train_dataloader = DataLoader(fine_dataset,batch_size=16,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16)


encoder = Encoder()
projection = Projection()
fully_connected = FullyConnected()
if os.path.exists('ssl/models/pre_trained_encoder_crop_flip_color_jitter.pth'):
    encoder.load_state_dict(torch.load('ssl/models/pre_trained_encoder_crop_flip_color_jitter.pth'))
    print("Loaded Encoder")
else:
    pre_training(encoder,projection)
fine_dataset = HsiDataset(mode="fine")
test_dataset = HsiDataset(mode="test")
fine_dataloader = DataLoader(fine_dataset,batch_size=16,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=16)
model = HSI_CNN(encoder,fully_connected)
print(model)
optimizer = Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
best_acc = 0
acc_list = [0]
for epoch in range(50):
    model.train()
    print(f'Epoch: {epoch}')
    for i,(X,y) in enumerate(fine_dataloader):
        logits = model(X)
        batch_loss = criterion(logits,y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if i%100 ==0:
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
            # print(f'Batch Accuracy: {acc} batch_id: {i}')
        if (epoch_acc/(i+1)) > best_acc:
            torch.save(model.state_dict(),'ssl/models/best_model.pth')
        print(f'Epoch Accuracy: {epoch_acc/(i+1)}')
        acc_list.append(epoch_acc/(i+1))
    
epoch_list = list(range(0,51))
plt.plot(epoch_list,acc_list)
plt.show()

        
