import pandas as pd
import torch
import numpy as np
import sys

from sklearn.model_selection import StratifiedKFold

import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

import progressbar

def create_data(file_path,sample_tag=False):
    if "macau" in file_path:
        if sample_tag:
            N_samples=20 #5 in total, with the mean included.
            N_dim=file_path[-3:-1]
            latent_pat=create_macau_sample_data(N_dim=int(N_dim),N_samples=N_samples,file_path=file_path)
        else:
            latent_pat=np.load(file_path+"mean_pat_latent.npy").T
            N_samples=1
    else:
        latent_pat=torch.load(file_path+"best_model.pt")["pat_lat.weight"].cpu().numpy() #latents without covariates
    #print(latent_pat.shape)
        covariates=pd.read_csv("~/Data/MIMIC/complete_covariates.csv").as_matrix() #covariates
        beta_u=torch.load(file_path+"best_model.pt")["beta_u"].cpu().numpy() #Coeffs for covariates
        latent_pat=np.dot(covariates[:,1:],beta_u)
        N_samples=1

    tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")
    tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]
    N_pat=tag_mat.shape[0]
    if sample_tag: #repeat the chain N_samples times
        tag_mat=np.tile(tag_mat,N_samples)
    print(tag_mat.shape)
    print(latent_pat.shape)
    return latent_pat,tag_mat,N_pat,N_samples

def create_macau_sample_data(N_dim,N_samples,file_path):
    #container=np.empty((N_pat*N_samples,N_dim,))
    container=np.load(file_path+"mean_pat_latent.npy").T #We keep the main in the samples
    print("Loading samples")
    for n in progressbar.progressbar(range(1,N_samples)):
        #container[((n-1)*N_pat):(n*N_pat),:]=np.loadtxt(dir_path+"-sample%d-U1-latents.csv"%n,delimiter=",").T
        container=np.append(container,np.loadtxt(file_path+str(N_dim)+"_macau"+"-sample%d-U1-latents.csv"%n,delimiter=",").T,axis=0)
    return(container)

class MLP_class_mod(nn.Module):
    def __init__(self,input_dim):
        super(MLP_class_mod,self).__init__()
        self.layer_1=nn.Linear(input_dim,200)
        self.layer_1bis=nn.Linear(200,70)
        self.layer_2=nn.Linear(70,20)
        self.layer_3=nn.Linear(20,1)
    def fwd(self,x):
        out=F.relu(self.layer_1(x))
        out=F.relu(self.layer_1bis(out))
        out=F.relu(self.layer_2(out))
        out=F.sigmoid(self.layer_3(out)).squeeze(1)
        return(out)

class latent_dataset(Dataset):
    def __init__(self,latents,tags):
        self.latents=torch.Tensor(latents)
        self.tags=torch.Tensor(tags).float()
    def __len__(self):
        return(self.latents.shape[0])
    def __getitem__(self,idx):
        return([self.latents[idx,:],self.tags[idx]])
    def get_dim(self):
        return self.latents.shape[1]

def train_mod(model,dataloader,dataloader_val):
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01)#,weight_decay=0.002)
    criterion = nn.BCELoss()#

    epochs_num=100
    for epoch in range(epochs_num):
        total_loss=0
        for i_batch,sampled_batch in enumerate(dataloader):
            optimizer.zero_grad()
            pred=model.fwd(sampled_batch[0])
            loss=criterion(pred,sampled_batch[1])
            loss.backward()
            optimizer.step()
            total_loss+=loss

        with torch.no_grad():
            for i_batch_val,sampled_batch in enumerate(dataloader_val):
                val_pred=model.fwd(sampled_batch[0])
                val_loss=criterion(val_pred,sampled_batch[1])
        print("Loss for epoch "+str(epoch)+" = "+str(total_loss/(i_batch+1))+" with validation loss = "+str(val_loss))
    return(model)

def index_with_samples(index_original,N_samp,N_pat):
    index_full=list(index_original)
    for i in range(1,N_samp):
        index_full+=list(index_original+i*N_pat)
    return(np.array(index_full))


if __name__=="__main__":
    n_splits=10
    sample_tag=True
    file_path=sys.argv[1:][0]
    latent_pat,tag_mat,N_pat,N_samp=create_data(file_path,sample_tag=sample_tag)
    cv=StratifiedKFold(n_splits=n_splits)
    cv.split(latent_pat,tag_mat)

    print("Nombre patients")
    print(N_pat)
    print("Nombre samples")
    print(N_samp)
    auc_mean=0
    for index_train,index_test in cv.split(latent_pat[:N_pat,:],tag_mat[:N_pat]):

        print(len(index_train)+len(index_test))
        print(max(index_train))
        model=MLP_class_mod(input_dim=latent_pat.shape[1]).double()
        print(len(index_train))
        print("type of index train")
        print(type(index_train))
        if sample_tag:
            index_train=index_with_samples(index_train,N_samp,N_pat)
            #index_test=index_with_samples(index_test,N_samp,N_pat)
        print("Max of index_train")
        print(max(index_train))
        print(len(index_train))

        latent_data=latent_dataset(latent_pat[index_train,:],tag_mat[index_train])
        dataloader = DataLoader(latent_data, batch_size=len(index_train),shuffle=True,num_workers=2)
        latent_data_val=latent_dataset(latent_pat[index_test,:],tag_mat[index_test])
        dataloader_val = DataLoader(latent_data_val, batch_size=len(index_test),shuffle=True,num_workers=2)

        model=train_mod(model,dataloader,dataloader_val)

        val_data=torch.tensor((latent_pat[index_test,:]))
        print(type(val_data))
        pred_val=model.fwd(val_data[:])
        auc_roc=roc_auc_score(tag_mat[index_test],pred_val.detach().numpy())
        print(auc_roc)
        auc_mean+=auc_roc
    print("mean AUC for this setting : "+str(auc_mean/n_splits))
