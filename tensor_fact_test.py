
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from tensor_fact import tensor_fact, TensorFactDataset

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import sys

import matplotlib
import matplotlib.pyplot as plt

class TensorTestDataset(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor_val.csv",file_path="~/Documents/Data/Full_MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
        #We want a dataframe with the each row being a different patient
        df_single=self.lab_short.drop_duplicates(["UNIQUE_ID"])[["UNIQUE_ID"]+cov_values]
        self.tensor_mat=df_single.as_matrix()
        #print(self.lab_short.dtypes)
    def __len__(self):
        return self.pat_num
    def __getitem__(self,idx):
        #print(self.lab_short["VALUENUM"].iloc[idx].values)
        #return([torch.from_numpy(self.lab_short.iloc[idx][["UNIQUE_ID","LABEL_CODE","TIME_STAMP"]].astype('int64').as_matrix()),torch.from_numpy(self.lab_short.iloc[idx][self.cov_values].as_matrix()),torch.from_numpy(self.lab_short.iloc[idx][self.time_values].astype('float64').as_matrix()),torch.tensor(self.lab_short["VALUENUM"].iloc[idx],dtype=torch.double)])
        #return(self.lab_short.iloc[idx].as_matrix())
        return(self.tensor_mat[idx,:])

def load_current_model(directory="~/Projects/Tensor_Fact/trained_models/8_dim_500epochs_lr02/"): #Function to load the saved model.
    train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train.csv",file_path="~/Data/MIMIC/")
    mod=tensor_fact(n_pat=train_dataset.pat_num,n_meas=30,n_t=101,l_dim=8,n_u=18,n_w=1)
    mod.double()
    mod.load_state_dict(torch.load(directory+"current_model.pt"))
    return(mod)

def main():
    first_pass=True
    pat_change="n"
    while(True):
        mod=load_current_model()
        criterion=nn.MSELoss()
        val_dataset=TensorTestDataset(csv_file_serie="lab_short_tensor_val.csv")
        train_data=pd.read_csv("~/Documents/Data/Full_MIMIC/lab_short_tensor_train.csv")
        val_data=pd.read_csv("~/Documents/Data/Full_MIMIC/lab_short_tensor_val.csv")

        if (not first_pass):
            pat_change=input("Do you want to change the patient index ? [y/n]")

        if (pat_change=="y" or first_pass):
            pat_idx=int(input("Enter the index of the patient you want to plot :"))
            assert(isinstance(pat_idx,int))
            pat_sample=torch.tensor(val_dataset[pat_idx]).double()
            preds=mod.forward_full(pat_sample[0].to(torch.long),pat_sample[1:].unsqueeze(0))

        feat_idx=int(input("Enter the index of the feature you want to plot :"))
        assert(isinstance(feat_idx,int))

        targets=train_data.loc[(train_data["UNIQUE_ID"]==pat_sample[0])&(train_data["LABEL_CODE"]==feat_idx)][["TIME_STAMP","VALUENORM"]].values
        targets_val=val_data.loc[(val_data["UNIQUE_ID"]==pat_sample[0])&(val_data["LABEL_CODE"]==feat_idx)][["TIME_STAMP","VALUENORM"]].values
        plt.plot(preds.detach().numpy()[:,feat_idx,:].T,label="Inferred curve")
        plt.scatter(targets[:,0],targets[:,1],label="Training Data")
        plt.scatter(targets_val[:,0],targets_val[:,1],label="Validation Data")
        plt.legend()
        plt.show()

        first_pass=False

def plot_latent_times():
    mod=load_current_model()
    plt.plot(mod.time_lat.weight.detach().numpy())
    plt.show()
    print(mod.pat_lat.weight)

if __name__=="__main__":
    plot_latent_times()
