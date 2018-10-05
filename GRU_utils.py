import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable


class GRU_teach(nn.Module):
    def __init__(self,device,x_dim,y_dim,latents=100):
        super(GRU_teach,self).__init__()
        self.device=device

        self.beta_mu_layer=nn.Linear(x_dim,100)
        self.beta_mu_layer2=nn.Linear(100,latents)
        self.beta_sigma_layer=nn.Linear(x_dim,100)
        self.beta_sigma_layer2=nn.Linear(100,latents)

        self.GRU1=nn.GRUCell(2*y_dim,latents)
        self.layer1=nn.Linear(latents,latents)
        self.layer2=nn.Linear(latents,y_dim)

        self.classif_layer1=nn.Linear(latents,50)
        self.classif_layer2=nn.Linear(50,1)
    def forward(self,x,y,y_mask):
        #x are the covariates and y are the observed samples with the missing mask
        #Dims are batch X input_dim X T for y
        #         batch X x_dim for x
        #y_mask is the OBSERVED mask (1 if sample is observed)
        #h_t=self.beta_layer(x)
        mu_0=self.beta_mu_layer2(F.tanh(self.beta_mu_layer(x)))
        log_std=self.beta_sigma_layer2(F.tanh(self.beta_sigma_layer(x)))

        h_t=self.reparametrize(mu_0,log_std)

        y_input=torch.zeros(y.size(0),y.size(1)).to(self.device)
        output=torch.zeros(y.size()).to(self.device) #tensor of output samples.
        for t in range(y.size(2)):
            y_input[y_mask[:,:,t]]=y[:,:,t][y_mask[:,:,t]]
            y_interleaved=torch.stack((y_input,y_mask[:,:,t].float()),dim=2).view(y.size(0),2*y.size(1))
            h_t =self.GRU1(y_interleaved,h_t)
            output[:,:,t]=self.layer2(F.tanh(self.layer1(h_t)))
            y_input=output[:,:,t]
        #Classification task.
        out_class=F.relu(self.classif_layer1(h_t))
        out_class=F.sigmoid(self.classif_layer2(out_class)).squeeze(1)
        return [output,out_class]

    def reparametrize(self,mu,logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


class GRU_teach_dataset(Dataset):
    def __init__(self,csv_file_serie="LSTM_tensor_train.csv",file_path="~/Documents/Data/Full_MIMIC/Clean_data/",cov_path="LSTM_covariates_train.csv",tag_path="LSTM_death_tags_train.csv"):
        #Create a tensor whose missing entries (0) are NaNs.
        data_short=pd.read_csv(file_path+csv_file_serie)
        d_idx=dict(zip(data_short["UNIQUE_ID"].unique(),np.arange(data_short["UNIQUE_ID"].nunique())))
        data_short["PATIENT_IDX"]=data_short["UNIQUE_ID"].map(d_idx)

        idx_mat=data_short[["PATIENT_IDX","LABEL_CODE","TIME_STAMP","VALUENORM"]].values

        idx_tens=torch.LongTensor(idx_mat[:,:-1])
        val_tens=torch.Tensor(idx_mat[:,-1])
        sparse_data=torch.sparse.FloatTensor(idx_tens.t(),val_tens)
        self.data_matrix=sparse_data.to_dense()

        self.data_matrix[self.data_matrix==0]=torch.tensor([np.nan]) #Done

        #Create the observation mask
        self.observed_mask=(self.data_matrix==self.data_matrix)

        #The covariates
        df_cov=pd.read_csv(file_path+cov_path)
        df_cov["PATIENT_IDX"]=df_cov["UNIQUE_ID"].map(d_idx)
        df_cov.set_index("PATIENT_IDX",inplace=True)
        df_cov.sort_index(inplace=True)
        self.cov_u=torch.tensor(df_cov.values[:,1:]).to(torch.float)

        #Death tags
        tags_df=pd.read_csv(file_path+tag_path)
        tags_df["PATIENT_IDX"]=tags_df["UNIQUE_ID"].map(d_idx)
        tags_df.sort_values(by="PATIENT_IDX",inplace=True)
        self.tags=torch.Tensor(tags_df["DEATHTAG"].values).float()

        assert(self.tags.size(0)==self.cov_u.size(0))
    def __len__(self):
        return self.data_matrix.size(0)
    def __getitem__(self,idx):
        return([self.data_matrix[idx,:,:],self.observed_mask[idx,:,:],self.cov_u[idx,:],self.tags[idx]])
