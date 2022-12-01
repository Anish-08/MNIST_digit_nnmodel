import random
import numpy as np
import h5py
import torch
import torch.nn as nn


class Solve(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.lay1 = nn.Linear(i,128)
        self.lay2 = nn.Linear(128, 16)
        self.lay3 = nn.Linear(16,o)
        self.activ = nn.ReLU()

    def forward(self, v):
        u = self.activ(self.lay1(v))
        u = self.activ(self.lay2(u))
        u = self.lay3(u)
        return u



def main():

    f = h5py.File('mnist.mat','r')

    labels_train = np.array(f["labels_train"])
    y_tot = torch.from_numpy(labels_train)
    y_tot = torch.reshape(y_tot, [60000])

    digits_train = np.array(f["digits_train"])
    t = torch.from_numpy(digits_train)
    t = t.float()
    x_tot = torch.reshape(t, [60000,784])
    #print(x)
    #print(y)


    ll = nn.CrossEntropyLoss()
    My_model = Solve(784,10)
    op = torch.optim.Adam(My_model.parameters(),lr = 0.00001)
    
    try:
        for s in range(500000):
            p = torch.tensor(random.sample(range(1,60000), 100))
            y = torch.index_select(y_tot,0,p)
            x = torch.index_select(x_tot,0, p)
            x.requires_grad = True 
            y_new = My_model(x)
            l = ll(y_new,y)
            l.backward()
            op.step()
            op.zero_grad()
            if (s+1)%50000==0 :
                print(l.item())
    except KeyboardInterrupt:
        pass 
    

    torch.save(My_model.state_dict(), "Digits_Model_3")


if __name__=='__main__':
    main()
