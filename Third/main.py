import random
import numpy as np
import h5py
import torch
import torch.nn as nn


class Solve(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv_lay = nn.Conv2d(32,32,kernel_size=5)
        self.lay1 = nn.Linear(24*24,128)
        self.lay2 = nn.Linear(128, 16)
        self.lay3 = nn.Linear(16,o)
        self.activ = nn.ReLU()

    def forward(self, v):
        #print(v.shape)
        x = torch.reshape(v,[32,28,28])
        x = self.conv_lay(x)
        #print(x.shape)
        u = torch.reshape(x,[32,24*24])
        u = self.activ(u)
        u = self.activ(self.lay1(u))
        u = self.activ(self.lay2(u))
        u = self.lay3(u)
        return u



def main():

    f = h5py.File('../mnist.mat','r')

    labels_train = np.array(f["labels_train"])
    labels_train = labels_train.reshape((60000,1))

    digits_train = np.array(f["digits_train"])
    digits_train = digits_train.reshape((60000,784))
   
    data = torch.from_numpy(np.hstack((digits_train,labels_train)))
    print(labels_train.shape)
    print(digits_train.shape)
    print(data.shape)
    

    data_set = torch.utils.data.DataLoader(data,batch_size = 32, shuffle = True)


    ll = nn.CrossEntropyLoss()
    My_model = Solve(784,10)
    op = torch.optim.Adam(My_model.parameters(),lr = 0.0001)
    
    try:
        for r in range(100):
            for t in data_set:
                y = t[:,784]
                x = t[:,:784].float()
                x.requires_grad = True 
                y_new = My_model(x)
                l = ll(y_new,y)
                l.backward()
                op.step()
                op.zero_grad()
            if r%10==0:
                print(l.item())
    except KeyboardInterrupt:
        pass 
    

    torch.save(My_model.state_dict(), "Digits_Model_2")


if __name__=='__main__':
    main()
