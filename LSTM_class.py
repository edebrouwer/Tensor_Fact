#LSTM Classifier on the latent/observed time series.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Sequence(nn.Module):
    def __init__(self,input_dim):
        print("Model Initialization")
        super(Sequence, self).__init__()
        self.input_dim=input_dim
        self.lstm1 = nn.LSTM(input_dim, 20)
        self.class_layer1=nn.Linear(20,20)
        self.class_layer2=nn.Linear(20,1)

    def fwd(self,data_in,batch_dim):

        h_t = torch.zeros(1,batch_dim, 20, dtype=torch.double)
        c_t = torch.zeros(1,batch_dim, 20, dtype=torch.double)
        #Data should be in the lengthxbatchxdim
        _,(h_n,c_n)=self.lstm1(data_in,(h_t,c_t))

        out=F.relu(self.class_layer1(h_n))
        out=F.sigmoid(self.class_layer2(out))

        return out

def dummy_data_gen(time_steps,batch_size,in_size):
    dat=torch.zeros(time_steps,batch_size,in_size)
    label=torch.zeros(batch_size)
    for i in range(batch_size):
        label[i]=int(np.random.binomial(1,0.5))
        a=torch.sin((1+label[i])*torch.linspace(0,1,time_steps))
        dat[:,i,:]=torch.sin((1+label[i])*torch.linspace(0,1,time_steps))
    return(dat)

def load_true_data(file_path):
    if "macau" in file_path:
        latent_pat=np.load(file_path+"mean_pat_latent.npy").T

if __name__=="__main__":
    batch_size=4
    time_steps=10
    in_size=2
    dummy_data=torch.tensor(torch.randn(time_steps,batch_size, in_size),dtype=torch.double)
    mod=Sequence(in_size)
    mod.double()
    res=mod.fwd(dummy_data,batch_size)
    print(res.size())
    print(res)
    dummy_data_gen(time_steps,batch_size, 1)
