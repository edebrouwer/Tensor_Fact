
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Check Ethan Rosenthal github

class tensor_fact(nn.Module):
    def __init__(self,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3):
        super(tensor_fact,self).__init__()
        self.n_pat=n_pat
        self.n_meas=n_meas
        self.n_t=n_t
        self.l_dim=l_dim
        self.n_u=n_u
        self.n_w=n_w
        self.pat_lat=nn.Embedding(n_pat,l_dim).double() #sparse gradients ?
        self.meas_lat=nn.Embedding(n_meas,l_dim).double()
        self.time_lat=nn.Embedding(n_t,l_dim).double()
        self.beta_u=torch.zeros([n_u,l_dim],requires_grad=True).double()
        self.beta_w=torch.zeros([n_w,l_dim],requires_grad=True).double()
    def forward(self,idx_pat,idx_meas,idx_t,cov_u,cov_w):
        pred=((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)


class TensorFactDataset(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor.csv",file_path="~/Documents/Data/Full_MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.length=len(self.lab_short.index)
        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        self.cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
        self.time_values=["TIME_STAMP","TIME_SQ"]
        #print(self.lab_short.dtypes)
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        #print(self.lab_short["VALUENUM"].iloc[idx].values)
        return([torch.from_numpy(self.lab_short.iloc[idx][["UNIQUE_ID","LABEL_CODE","TIME_STAMP"]].astype('int64').as_matrix()),torch.from_numpy(self.lab_short.iloc[idx][self.cov_values].as_matrix()),torch.from_numpy(self.lab_short.iloc[idx][self.time_values].astype('float64').as_matrix()),torch.tensor(self.lab_short["VALUENUM"].iloc[idx],dtype=torch.double)])


def main():
    #With Adam optimizer


    import time



    tens_dataset=TensorFactDataset()
    dataloader = DataLoader(tens_dataset, batch_size=1000,shuffle=True)

    mod=tensor_fact(n_pat=tens_dataset.pat_num,n_meas=30,n_t=101,n_u=18,n_w=2)
    #mod.double()

    optimizer=torch.optim.Adam(mod.parameters(), lr=0.001)
    criterion = nn.MSELoss(size_average=False)#
    epochs_num=1

    for epoch in range(epochs_num):
        print("EPOCH : "+str(epoch))
        for i_batch,sampled_batch in enumerate(dataloader):

            print(i_batch)
            starttime=time.time()

            optimizer.zero_grad()
            preds=mod.forward(sampled_batch[0][:,0],sampled_batch[0][:,1],sampled_batch[0][:,2],sampled_batch[1],sampled_batch[2])
            loss=criterion(preds,sampled_batch[3])
            loss.backward()
            optimizer.step()
            print(time.time()-starttime)

if __name__=="__main__":
    main()
