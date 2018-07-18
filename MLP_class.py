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


def create_data(file_path):
    if "macau" in file_path:
        latent_pat=np.load(file_path+"mean_pat_latent.npy").T
    else:
        latent_pat=torch.load(file_path+"best_model.pt")["pat_lat.weight"].cpu().numpy() #latents without covariates
    #print(latent_pat.shape)
        covariates=pd.read_csv("~/Data/MIMIC/complete_covariates.csv").as_matrix() #covariates
        beta_u=torch.load(file_path+"best_model.pt")["beta_u"].cpu().numpy() #Coeffs for covariates
        latent_pat=np.dot(covariates[:,1:],beta_u)

    tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")
    tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]
    print(tag_mat.shape)
    print(latent_pat.shape)
    return latent_pat,tag_mat

class MLP_class_mod(nn.Module):
    def __init__(self,input_dim):
        super(MLP_class_mod,self).__init__()
        self.layer_1=nn.Linear(input_dim,50)
        self.layer_1bis=nn.Linear(50,50)
        self.layer_2=nn.Linear(50,20)
        self.layer_3=nn.Linear(20,1)
    def fwd(self,x):
        out=F.relu(self.layer_1(x))
        out=F.relu(self.layer_1bis(out))
        out=F.relu(self.layer_2(out))
        out=F.sigmoid(self.layer_3(out)).squeeze(1)
        return(out)

class latent_dataset(Dataset):
    def __init__(self,latents,tags):
        self.latents=latents
        self.tags=tags.astype(float)
    def __len__(self):
        return(self.latents.shape[0])
    def __getitem__(self,idx):
        return([self.latents[idx,:],self.tags[idx]])

def train_mod(model,dataloader):
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0)
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
        print("Loss for epoch "+str(epoch)+" = "+str(total_loss/(i_batch+1)))
    return(model)


cv=StratifiedKFold(n_splits=10)
cv.split(latent_pat,tag_mat)


if __name__=="__main__":
    file_path=sys.argv[1:][0]
    latent_pat,tag_mat=create_data(file_path)
    cv=StratifiedKFold(n_splits=10)
    cv.split(latent_pat,tag_mat)

    for index_train,index_test in cv.split(latent_pat):

        model=MLP_class_mod(input_dim=len(index_train)).double()
        latent_data=latent_dataset(latent_pat[index_train,:],tag_mat[index_train])
        dataloader = DataLoader(latent_data, batch_size=10000,shuffle=True,num_workers=2)
        model=train_mod(model,dataloader)

        pred_val=model.fwd(latent_pat[index_test,:])
        auc_roc=roc_auc_score(tag_mat[index_test],pred_val)
        print(auc_roc)
