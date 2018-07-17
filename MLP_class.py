
import torch
import numpy as np
import sys

import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


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


class MLP_class_mod(nn.Module):
    def __init__(self,input_dim):
        super(MLP_class_mod,self).__init__()
        self.layer_1=nn.Linear(input_dim,20)
        self.layer_2=nn.Linear(20,20)
        self.layer_3=nn.Linear(20,1)
    def fwd(self,x):
        out=F.relu(self.layer_1(x))
        out=F.relu(self.layer_2(out))
        out=self.layer_3(out).squeeze(1)

class latent_dataset(Dataset):
    def __init__(self,latents,tags):
        self.latents=latents
        self.tags=tags
    def __len__(self):
        return(self.latents.size(0))
    def __getitem__(self,idx):
        return([self.latents[idx,:],self.tags[idx]])

def train_mod(model,dataloader):
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0)
    criterion = nn.BCELoss()#

    epochs_num=10
    for epoch in epochs_num:
        for i_batch,sampled_batch in enumerate(dataloader):
            optimizer.zero_grad()
            pred=model.fwd(sampled_batch[0])
            loss=criterion(pred,sampled_batch[1])
            loss.backward()
            optimizer.step()
    return(model)


if __name__=="__main__":
    file_path=sys.argv[1:][0]
    latent_pat,tag_mat=create_data(file_path)
    model=MLP_class_mod(input_dim=laten_pat.size(1))
    dataloader = DataLoader(latent_dataset, batch_size=1000,shuffle=True,num_workers=2)
    model=train_mod(model,dataloader)
