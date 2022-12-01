import numpy as np
import h5py
import torch
import torch.nn as nn
from main import Solve

def main():
    SAMP = 1000
    f = h5py.File('../mnist.mat','r')

    labels_train = np.array(f["labels_test"])
    siz = 10000
    y = torch.from_numpy(labels_train)
    y = torch.reshape(y, [siz])

    digits_train = np.array(f["digits_test"])
    t = torch.from_numpy(digits_train)
    t = t.float()
    x = torch.reshape(t, [siz,784])
    

    testmodel = Solve(784,10)
    testmodel.load_state_dict(torch.load('Digits_Model'))
    testmodel.eval()

    sum = 0
    for i in range(siz):
        samp = x[i]
        n = torch.argmax(testmodel(samp)).item()
        if n==labels_train[0][i]:
            sum = sum +1

    print(sum*100/siz)


if __name__=='__main__':
    main()
