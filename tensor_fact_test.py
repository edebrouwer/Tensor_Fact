
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from tensor_fact import tensor_fact, TensorFactDataset

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def load_current_model(): #Function to load the saved model.
       
        train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train.csv")
        mod=tensor_fact(n_pat=train_dataset.pat_num,n_meas=30,n_t=101,l_dim=8,n_u=18,n_w=1)
        mod.double()
        mod.load_state_dict(torch.load("current_model.pt"))
        return(mod)
#if __name__=="main"
mod=load_current_model()
criterion=nn.MSELoss()
val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val.csv")
val_loader=DataLoader(val_dataset,batch_size=len(val_dataset))
with torch.no_grad():
    for i_batch, sampled_batch in enumerate(val_loader):
        indexes=sampled_batch[:,1:4].to(torch.long)
        cov_u=sampled_batch[:,4:22]
        cov_w=sampled_batch[:,3].unsqueeze(1)
        target=sampled_batch[:,-1]
        preds=mod.forward(indexes[:,0],indexes[:,1],indexes[:,2],cov_u,cov_w)
        loss=criterion(preds,target)
        print(loss)
        torch.save(preds,"validation_preds.pt")
