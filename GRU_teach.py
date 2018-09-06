import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import ray
import ray.tune as tune
from ray.tune.async_hyperband import AsyncHyperBandScheduler
#from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler
from ray.tune import Trainable, TrainingResult
from ray.tune.util import pin_in_object_store, get_pinned_object


class GRU_teach(nn.Module):
    def __init__(self,x_dim,y_dim,latents=100):
        super(GRU_teach,self).__init__()
        self.beta_layer=nn.Linear(x_dim,latents)
        self.GRU1=nn.GRUCell(2*y_dim,latents)
        self.layer1=nn.Linear(latents,y_dim)
    def forward(self,x,y,y_mask):
        #x are the covariates and y are the observed samples with the missing mask
        #Dims are batch X input_dim X T for y
        #         batch X x_dim for x
        #y_mask is the OBSERVED mask (1 if sample is observed)
        h_t=self.beta_layer(x)
        y_input=torch.zeros(y.size(0),y.size(1))
        output=torch.zeros(y.size()) #tensor of output samples.
        for t in range(y.size(2)):
            y_input[y_mask[:,:,t]]=y[:,:,t][y_mask[:,:,t]]
            y_interleaved=torch.stack((y_input,y_mask[:,:,t].float()),dim=2).view(y.size(0),2*y.size(1))
            h_t =self.GRU1(y_interleaved,h_t)
            output[:,:,t]=self.layer1(h_t)
            y_input=output[:,:,t]
        return output

class GRU_teach_dataset(Dataset):
    def __init__(self,csv_file_serie="LSTM_tensor_train.csv",file_path="~/Documents/Data/Full_MIMIC/Clean_data/",cov_path="LSTM_covariates_train.csv"):
        #Create a tensor whose missing entries (0) are NaNs.
        data_short=pd.read_csv(file_path+csv_file_serie)
        d_idx=dict(zip(data_short["UNIQUE_ID"].unique(),np.arange(data_short["UNIQUE_ID"].nunique())))
        data_short["PATIENT_IDX"]=data_short["UNIQUE_ID"].map(d_idx)

        idx_mat=data_short[["PATIENT_IDX","LABEL_CODE","TIME_STAMP","VALUENORM"]].as_matrix()

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
        self.cov_u=torch.tensor(df_cov.as_matrix()[:,1:]).to(torch.float)

    def __len__(self):
        return self.data_matrix.size(0)
    def __getitem__(self,idx):
        return([self.data_matrix[idx,:,:],self.observed_mask[idx,:,:],self.cov_u[idx,:]])

def train():
    data_train=GRU_teach_dataset()
    data_val=GRU_teach_dataset(csv_file_serie="LSTM_tensor_val.csv",cov_path="LSTM_covariates_val.csv")
    dataloader=DataLoader(data_train,batch_size=5000,shuffle=True,num_workers=2)
    mod=GRU_teach(data_train.cov_u.size(1),data_train.data_matrix.size(1))
    mod.float()

    optimizer=torch.optim.Adam(mod.parameters(),lr=0.005,weight_decay=0.00005)
    criterion=nn.MSELoss(reduce=False,size_average=False)
    for epoch in range(100):
        for i_batch, sampled_batch in enumerate(dataloader):

            optimizer.zero_grad()

            preds=mod.forward(sampled_batch[2],sampled_batch[0],sampled_batch[1])

            targets=sampled_batch[0]
            targets.masked_fill_(1-sampled_batch[1],0)
            preds.masked_fill_(1-sampled_batch[1],0)

            loss=torch.sum(criterion(preds,targets))/torch.sum(sampled_batch[1]).float()

            loss.backward()
            optimizer.step()
            print(loss.detach().numpy())

        with torch.no_grad():
            preds=mod.forward(data_val.cov_u,data_val.data_matrix,data_val.observed_mask)
            targets=data_val.data_matrix
            targets.masked_fill_(1-data_val.observed_mask,0)
            preds.masked_fill_(1-data_val.observed_mask,0)
            loss_val=torch.sum(criterion(preds,targets))/torch.sum(data_val.observed_mask).float()
            print("Validation Loss")
            print(loss_val)

class train_class(Trainable):
    def _setup(self):
        self.device=torch.device("cuda:0")
        #means_vec for imputation.

        mod=GRU_teach(get_pinned_object(data_train).cov_u.size(1),get_pinned_object(data_train).data_matrix.size(1))
        self.mod.float()
        self.mod.to(self.device)

        self.dataloader=DataLoader(get_pinned_object(data_train),batch_size=5000,shuffle=True,num_workers=2)

        self.timestep=0
    def _train(self):
        self.timestep+=1

        #Select learning rate depending on the epoch.
        if self.timestep<50:
            l_r=0.0005
        elif self.timestep<95:
            l_r=0.00015
        else:
            l_r=0.00005

        optimizer = torch.optim.Adam(self.mod.parameters(), lr=l_r, weight_decay=self.config["L2"])

        criterion=nn.MSELoss(reduce=False,size_average=False)

        for i_batch,sampled_batch in enumerate(self.dataloader):
            optimizer.zero_grad()
            preds=self.mod.forward(sampled_batch[2],sampled_batch[0],sampled_batch[1])
            targets=sampled_batch[0]
            targets.masked_fill_(1-sampled_batch[1],0)
            preds.masked_fill_(1-sampled_batch[1],0)
            loss=torch.sum(criterion(preds,targets))/torch.sum(sampled_batch[1]).float()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds=self.mod.forward(get_pinned_object(data_val).cov_u,get_pinned_object(data_val).data_matrix,get_pinned_object(data_val).observed_mask)
            targets=get_pinned_object(data_val).data_matrix
            targets.masked_fill_(1-get_pinned_object(data_val).observed_mask,0)
            preds.masked_fill_(1-get_pinned_object(data_val).observed_mask,0)
            loss_val=torch.sum(criterion(preds,targets))/torch.sum(get_pinned_object(data_val).observed_mask).float()
            print("Validation Loss")
            print(loss_val)

        return TrainingResult(mean_loss=loss_val,timesteps_this_iter=1)

    def _save(self,checkpoint_dir):
        path=os.path.join(checkpoint_dir,"checkpoint")
        torch.save(self.mod.state_dict(),path)
        np.save(path+"_timestep.npy",self.timestep)
        return path
    def _restore(self,checkpoint_path):
        self.mod.load_state_dict(torch.load(checkpoint_path))
        self.timestep=np.load(checkpoint_path+"_timestep.npy").item()


if __name__=="__main__":

    #train()
    ray.init(num_cpus=10,num_gpus=2)
    data_train=pin_in_object_store(GRU_teach_dataset(file_path="~/Data/MIMIC/"))
    data_val=pin_in_object_store(GRU_teach_dataset(file_path="~/Data/MIMIC/",csv_file_serie="LSTM_tensor_val.csv",cov_path="LSTM_covariates_val.csv"))

    tune.register_trainable("my_class", train_class)

    hyperband=AsyncHyperBandScheduler(time_attr="training_iteration",reward_attr="neg_mean_loss",max_t=50,grace_period=15)

    exp={
            'run':"my_class",
            'repeat':10,
            'stop':{"training_iteration":50},
            'trial_resources':{
                            "gpu":1,
                            "cpu":1
                        },
            'config':{
            "L2":lambda spec: 10**(3*random.random()-6)
        }
     }


    tune.run_experiments({"GRU_simple_2layersclassif_350epochs":exp},scheduler=hyperband)
