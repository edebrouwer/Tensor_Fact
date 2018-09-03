import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

import random
import ray
import ray.tune as tune
from ray.tune.async_hyperband import AsyncHyperBandScheduler
#from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler
from ray.tune import Trainable, TrainingResult
from ray.tune.util import pin_in_object_store, get_pinned_object
import os

parser=argparse.ArgumentParser(description="Baselines for TS classification")

#Model parameters
parser.add_argument('--L2',default=0,type=float,help="L2 penalty (weight decay")
parser.add_argument('--maxepochs',default=0,type=float,help="Max number of epochs for the training.")

#Model selection
parser.add_argument('--unique',action='store_true',help="Train a unique model and saves it.")



class GRU_mod(nn.Module):
    def __init__(self,input_dim,latents=100):
        #mean_feats should be a torch vector containing the computed means of each features for imputation.
        super(GRU_mean,self).__init__()

        self.layer1=nn.GRU(self.input_dim,latents,1,batch_first=True)
        self.classif_layer1=nn.Linear(latents,100)
        self.classif_layer1bis=nn.Linear(100,100)
        self.classif_layer2=nn.Linear(100,1)
    def forward(self,x):
        #x is a batch X  T x input_dim tensor
        out,h_n=self.layer1(x)
        pred=F.relu(self.classif_layer1(h_n))
        pred=F.relu(self.classif_layer1bis(pred))
        pred=F.sigmoid(self.classif_layer2(pred))
        return(pred[0,:,0])


class GRU_mean(nn.Module):
    def __init__(self,input_dim,mean_feats,device,latents=100,imputation_mode="mean"):
        #mean_feats should be a torch vector containing the computed means of each features for imputation.
        super(GRU_mean,self).__init__()
        if imputation_mode=="simple":
            self.imput="simple"
            self.input_dim=3*input_dim
        elif imputation_mode=="mean":
            self.imput="mean"
            self.input_dim=input_dim
        else:
            raise ValueError("Wrong imputation mode")

        self.mean_feats=mean_feats
        self.layer1=nn.GRU(self.input_dim,latents,1,batch_first=True)
        self.classif_layer1=nn.Linear(latents,100)
        self.classif_layer1bis=nn.Linear(100,100)
        self.classif_layer2=nn.Linear(100,1)
        self.device=device

    def forward(self,x):
        #x is a batch X  T x input_dim tensor

        if self.imput=="mean":
            x=self.impute(x)
        elif self.imput=="simple":
            x=self.impute_simple(x)
        else:
            raise ValueError("Not a valid imputation option")
        out,h_n=self.layer1(x)
        pred=F.relu(self.classif_layer1(h_n))
        pred=F.relu(self.classif_layer1bis(pred))
        pred=F.sigmoid(self.classif_layer2(pred))
        return(pred[0,:,0])

    def impute(self,x):
        #x is a batch X T x input_dim tensor
        #Replace NANs by the corresponding mean_values.
        n_batch=x.size(0)
        n_t=x.size(1)
        x_mean=self.mean_feats.repeat(n_batch,n_t,1).to(self.device) #Tensor with only the means
        non_nan_mask=(x==x)
        x_mean[non_nan_mask]=x[non_nan_mask]
        return(x_mean)

    def impute_simple(self,x):
        n_batch=x.size(0)
        n_t=x.size(1)

        observed_mask=(x==x) #1 if observed, 0 otherwise
        Delta=torch.zeros(x.size(),device=self.device) #
        #print(Delta.dtype)
        #print(observed_mask.dtype)
        for idt in range(1,n_t):
            a=torch.zeros((n_batch,x.size(2)),device=self.device).masked_scatter_(1-observed_mask[:,idt-1,:],Delta[:,idt-1,:])
            #a=(1-observed_mask[:,idt-1,:])*Delta[:,idt-1,:]
            Delta[:,idt,:]=torch.ones((n_batch,x.size(2)),device=self.device)+a#(1-observed_mask[:,idt-1,:])*Delta[:,idt-1,:]
        return torch.cat((self.impute(x),observed_mask.float(),Delta),dim=2)


class LSTMDataset_ByPat(Dataset):
    def __init__(self,csv_file_serie="LSTM_tensor_train.csv",file_path="~/Data/MIMIC/",cov_path="LSTM_covariates_train",tag_path="LSTM_death_tags_train.csv",transform=None,latents_path=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        d_idx=dict(zip(self.lab_short["UNIQUE_ID"].unique(),np.arange(self.lab_short["UNIQUE_ID"].nunique())))
        self.lab_short["PATIENT_IDX"]=self.lab_short["UNIQUE_ID"].map(d_idx)

        idx_mat=self.lab_short[["PATIENT_IDX","LABEL_CODE","TIME_STAMP","VALUENORM"]].as_matrix()

        idx_tens=torch.LongTensor(idx_mat[:,:-1])
        val_tens=torch.Tensor(idx_mat[:,-1])
        sparse_data=torch.sparse.FloatTensor(idx_tens.t(),val_tens)
        self.data_matrix=sparse_data.to_dense()

        self.data_matrix[self.data_matrix==0]=torch.tensor([np.nan])

        #covariates
        df_cov=pd.read_csv(file_path+cov_path+".csv")
        df_cov["PATIENT_IDX"]=df_cov["UNIQUE_ID"].map(d_idx)
        df_cov.set_index("PATIENT_IDX",inplace=True)
        df_cov.sort_index(inplace=True)
        self.cov_u=torch.tensor(df_cov.as_matrix()[:,1:]).to(torch.double)

        #Death tags
        tags_df=pd.read_csv(file_path+tag_path)
        tags_df["PATIENT_IDX"]=tags_df["UNIQUE_ID"].map(d_idx)
        tags_df.sort_values(by="PATIENT_IDX",inplace=True)
        self.tags=tags_df["DEATHTAG"].as_matrix()

        #Imputation with CPD
        if latents_path is not None:
            latents_pat_mean=np.load(latents_path+"mean_pat_latent.npy")
            latents_feat_mean=np.load(latents_path+"mean_feat_latent.npy")
            latents_time_mean=np.load(latents_path+"mean_time_latent.npy")
            reconstructed_tensor=torch.Tensor(np.einsum('il,jl,kl->ijk',latents_pat_mean,latents_feat_mean,latents_times_mean))
            #To be continued....
        #Some dimensions.
        self.covu_num=self.cov_u.size(1)
        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        self.meas_num=self.lab_short["LABEL_CODE"].nunique()
        self.length=self.pat_num
    def __len__(self):
        return self.pat_num
    def __getitem__(self,idx):
        return([idx,self.data_matrix[idx,:,:],self.tags[idx]])#,self.train_tags[idx]])

def train(device,epoch_max,L2):

    data_train=LSTMDataset_ByPat(file_path="~/Data/MIMIC/")
    data_val=LSTMDataset_ByPat(csv_file_serie="LSTM_tensor_val.csv",file_path="~/Data/MIMIC/",cov_path="LSTM_covariates_val",tag_path="LSTM_death_tags_val.csv")

    #means_vec for imputation.
    means_df=pd.Series.from_csv("~/Documents/Data/Full_MIMIC/Clean_data/mean_features.csv")
    means_vec=torch.tensor(means_df.as_matrix(),dtype=tensor.float)

    dataloader=DataLoader(data_train,batch_size=5000,shuffle=True,num_workers=2)
    dataloader_val= DataLoader(data_val,batch_size=1000,shuffle=False)

    mod=GRU_mean(data_train.meas_num,means_vec,device,imputation_mode="simple")
    #mod.double()

    for epoch in range(epoch_max):
        if self.timestep<50:
            l_r=0.0005
        elif self.timestep<95:
            l_r=0.00015
        else:
            l_r=0.00005
        optimizer=torch.optim.Adam(mod.parameters(),lr=l_r,weight_decay=L2)
        optimizer.zero_grad()

        criterion=nn.BCELoss()


        for i_batch,sampled_batch in enumerate(dataloader):
            optimizer.zero_grad()
            target=sampled_batch[2].float().to(device)
            pred=mod.forward(torch.transpose(sampled_batch[1].to(device),1,2)) #Feed as batchxtimexfeatures
            loss=criterion(pred,target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss_val=0
            for i_val,batch_val in enumerate(dataloader_val):
                target=batch_val[2].float().to(device)
                pred=mod.forward(torch.transpose(batch_val[1].to(device),1,2)) #Feed as batchxtimexfeatures
                loss_val+=roc_auc_score(target,pred)
        auc_mean=loss_val/(i_val+1)
        print(auc_mean)
    outfile_path="./unique_model.pt"
    torch.save(mod.state_dict(),path)
    print("Model saved in the file "+outfile_path)

    #Compute validation AUC curve
    with torch.no_grad():
        fpr,tpr,_ = roc_curve(mod.forward(torch.transpose(data_val.data_matrix.to(device),1,2)).cpu().detach().numpy(),data_val.tags.numpy())
        roc_auc=auc(fpr,tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()




    print('Start training')
    for i_batch,sampled_batch in enumerate(dataloader):
        optimizer.zero_grad()
        target=sampled_batch[2].float()
        print(sampled_batch[1].size())
        pred=mod.forward(torch.transpose(sampled_batch[1],1,2)) #Feed as batchxtimexfeatures
        loss=criterion(pred,target)
        loss.backward()
        optimizer.step()
        print(loss.detach().numpy())


class train_class(Trainable):
    def _setup(self):
        self.device=torch.device("cuda:0")
        #means_vec for imputation.

        self.mod=GRU_mean(get_pinned_object(data_train).meas_num,get_pinned_object(means_vec),self.device,imputation_mode="simple")
        self.mod.float()
        self.mod.to(self.device)

        self.dataloader=DataLoader(get_pinned_object(data_train),batch_size=5000,shuffle=True,num_workers=2)
        self.dataloader_val= DataLoader(get_pinned_object(data_val),batch_size=1000,shuffle=False)

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

        criterion=nn.BCELoss()

        for i_batch,sampled_batch in enumerate(self.dataloader):
            optimizer.zero_grad()
            target=sampled_batch[2].float().to(self.device)
            pred=self.mod.forward(torch.transpose(sampled_batch[1].to(self.device),1,2)) #Feed as batchxtimexfeatures
            loss=criterion(pred,target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_val=0
            for i_val,batch_val in enumerate(self.dataloader_val):
                target=batch_val[2].float().to(self.device)
                pred=self.mod.forward(torch.transpose(batch_val[1].to(self.device),1,2)) #Feed as batchxtimexfeatures
                loss_val+=roc_auc_score(target,pred)
        auc_mean=loss_val/(i_val+1)

        return TrainingResult(mean_accuracy=auc_mean,timesteps_this_iter=1)

    def _save(self,checkpoint_dir):
        path=os.path.join(checkpoint_dir,"checkpoint")
        torch.save(self.mod.state_dict(),path)
        print("SAVIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIING")
        #raise Exception()
        #torch.cuda.empty_cache()
        np.save(path+"_timestep.npy",self.timestep)
        return path
    def _restore(self,checkpoint_path):
        print("LOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADING")
        self.mod.load_state_dict(torch.load(checkpoint_path))
        self.timestep=np.load(checkpoint_path+"_timestep.npy").item()

if __name__=="__main__":

    opt=parser.parse_args()
    if opt.unique:
        return(train("cuda:0",opt.maxepochs,opt.L2))
    else:
        ray.init(num_cpus=10,num_gpus=2)


        data_train=pin_in_object_store(LSTMDataset_ByPat(file_path="~/Data/MIMIC/"))
        data_val=pin_in_object_store(LSTMDataset_ByPat(csv_file_serie="LSTM_tensor_val.csv",file_path="~/Data/MIMIC/",cov_path="LSTM_covariates_val",tag_path="LSTM_death_tags_val.csv"))
        means_df=pd.Series.from_csv("~/Data/MIMIC/mean_features.csv")
        means_vec=pin_in_object_store(torch.tensor(means_df.as_matrix(),dtype=torch.float))

        tune.register_trainable("my_class", train_class)

        hyperband=AsyncHyperBandScheduler(time_attr="training_iteration",reward_attr="mean_accuracy",max_t=350,grace_period=15)

        exp={
                'run':"my_class",
                'repeat':30,
                'stop':{"training_iteration":350},
                'trial_resources':{
                                "gpu":1,
                                "cpu":1
                            },
                'config':{
                "L2":lambda spec: 10**(3*random.random()-6)
            }
         }


         tune.run_experiments({"GRU_simple_2layersclassif_350epochs":exp},scheduler=hyperband)
