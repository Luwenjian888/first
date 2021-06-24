# # -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
import os

class inverse_matrix():
    def __init__(self, epochs=500, lr=1e-3,batch_size=16,train_sample=20000):
        self.epochs = epochs
        self.lr = lr
        self.log=[]
        self.batch_size=batch_size
        self.train_sample=train_sample
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize_weights(self,size):
        self.w_inv=torch.randn((size[1],size[0]), device=self.device, dtype=torch.float, requires_grad=True)
        
    def fit(self, w):
        if torch.is_tensor(w):
            w=w
        else:
            w=torch.from_numpy(w)
        size= w.shape
        self.initialize_weights(size)
        
        x = torch.rand(self.train_sample,size[1])
        y = (w@x.T).T
        print('x shape:',x.shape,'y shape:',y.shape)
        
        optimizer= torch.optim.Adam([self.w_inv], lr=self.lr)
        loss_func= torch.nn.MSELoss(reduction='mean')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',verbose=True,patience=5)
        trainset= torch.utils.data.TensorDataset(y,x)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,shuffle=True)
        
        #train model
        start = time.process_time()
        print('start trainning!')
        for epoch in range(self.epochs):
            train_loss = 0.
            for batch_idx, sample in enumerate(train_loader):
                data, target = sample[0].to(self.device),sample[1].to(self.device)
                optimizer.zero_grad()
                output= (self.w_inv@data.T).T
                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss=train_loss/(batch_idx+1)
            self.log.append(train_loss)
            current_lr=optimizer.state_dict()['param_groups'][0]['lr']
            print('Train Epoch:{:4}  lr:{:.8f}:    train_loss: {:.6f} '.format(epoch+1,current_lr,train_loss))
            scheduler.step(train_loss)
        end = time.process_time()
        print('finish trainning!')
        print('trainning time:',end-start,' S')
        return self.w_inv.detach().numpy()
    def plot(self, log):
        plt.figure()
        plt.plot(log)
        plt.show()
    def evaluate(self, w,w_inv):
        if torch.is_tensor(w):
            w=w.numpy()
        else:
            w=w
        
        I=w_inv@w
        
        plt.figure()
        plt.imshow(I)
        plt.colorbar()
        plt.show()


if os.path.exists('w.npy'):
        w=np.load('w.npy')
        w_inv=np.load('w_inv.npy')
        I=w_inv@w
        
        plt.figure()
        plt.imshow(I)
        plt.colorbar()
        plt.show()
else:
        epochs=500
        batch_size=16
        lr=1e-3
        train_sample=20000
        
        w=torch.rand(32,32)
        
        model=inverse_matrix(epochs=epochs, lr=lr,batch_size=batch_size,train_sample=train_sample)
        w_inv=model.fit(w)
        model.plot(model.log)
        model.evaluate(w, w_inv)
        log=model.log
        
        np.save('w.npy',w.numpy())
        np.save('w_inv.npy',w_inv)
