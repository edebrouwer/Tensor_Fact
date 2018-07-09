
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import os

class tensor_fact(nn.Module):
    def __init__(self,device,covariates,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3,l_kernel=3,sig2_kernel=1):
        #mode can be Normal, Deep, XT , Class or  by_pat
        super(tensor_fact,self).__init__()
        self.n_pat=n_pat
        self.n_meas=n_meas
        self.n_t=n_t
        self.l_dim=l_dim
        self.n_u=n_u
        self.n_w=n_w
        self.pat_lat=nn.Embedding(n_pat,l_dim) #sparse gradients ?
        self.pat_lat.weight=nn.Parameter(0.05*torch.randn([n_pat,l_dim]))
        self.meas_lat=nn.Embedding(n_meas,l_dim)
        self.meas_lat.weight=nn.Parameter(0.05*torch.randn([n_meas,l_dim]))
        self.time_lat=nn.Embedding(n_t,l_dim)#.double()
        self.time_lat.weight=nn.Parameter(0.005*torch.randn([n_t,l_dim]))
        self.beta_u=nn.Parameter(torch.randn([n_u,l_dim],requires_grad=True))#.double())
        self.beta_w=nn.Parameter(torch.randn([n_w,l_dim],requires_grad=True))#.double())

        self.cov_w_fixed=torch.tensor(range(0,n_t),device=device,dtype=torch.double,requires_grad=False).unsqueeze(1)#.double()
        self.covariates_u=covariates.to(device)
        #self.covariates_u.load_state_dict({'weight':covariates})
        #self.covariates_u.weight.requires_grad=False

        full_dim=3*l_dim+n_u+n_w
        #print(full_dim)


        #if (mode=="Class"):
    #        #classification
    #        self.layer_class_1=nn.Linear((l_dim+n_u),20)
    #        self.layer_class_2=nn.Linear(20,1)


        #Kernel_computation
        #x_samp=np.linspace(0,(n_t-1),n_t)
        #SexpKernel=np.exp(-(np.array([x_samp]*n_t)-np.expand_dims(x_samp.T,axis=1))**2/(2*l_kernel**2))
        #SexpKernel[SexpKernel<0.1]=0
        #self.inv_Kernel=torch.tensor(np.linalg.inv(SexpKernel)/sig2_kernel,requires_grad=False)

    def forward(self,idx_pat,idx_meas,idx_t):
        pred=((self.pat_lat(idx_pat)+torch.mm(self.covariates_u[idx_pat,:],self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(self.cov_w_fixed[idx_t,:],self.beta_w))).sum(1)
        return(pred)

    def label_pred(self,idx_pat,cov_u): #Classifiction task
        merged_input=torch.cat((self.pat_lat(idx_pat),cov_u),1)
        out=F.relu(self.layer_class_1(merged_input))
        out=F.sigmoid(self.layer_class_2(out))
        return(out)
    def compute_regul(self):
        regul=torch.trace(torch.exp(-torch.mm(torch.mm(torch.t(self.time_lat.weight),self.inv_Kernel),self.time_lat.weight)))
        return(regul)



class deep_tensor_fact(tensor_fact):
    def __init__(self,device,covariates,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3,l_kernel=3,sig2_kernel=1):
        super(deep_tensor_fact,self).__init__(device=device,covariates=covariates,n_pat=n_pat,n_meas=n_meas,n_t=n_t,l_dim=l_dim,n_u=n_u,n_w=n_w,l_kernel=l_kernel,sig2_kernel=sig2_kernel)
        self.layer_1=nn.Linear(full_dim,50)
        self.layer_2=nn.Linear(50,50)
        self.layer_3=nn.Linear(50,20)
        self.last_layer=nn.Linear(20,1)
    def forward(self,idx_pat,idx_meas,idx_t):
        merged_input=torch.cat((self.pat_lat(idx_pat),self.meas_lat(idx_meas),self.time_lat(idx_t),self.covariates_u[idx_pat,:],self.cov_w_fixed[idx_t,:]),1)
        #print(merged_input.size())
        out=F.relu(self.layer_1(merged_input))
        out=F.relu(self.layer_2(out))
        out=F.relu(self.layer_3(out))
        out=self.last_layer(out).squeeze(1)
        return(out)

class XT_tensor_fact(tensor_fact):
    def __init__(self,device,covariates,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3,l_kernel=3,sig2_kernel=1):
        super(XT_tensor_fact,self).__init__(device=device,covariates=covariates,n_pat=n_pat,n_meas=n_meas,n_t=n_t,l_dim=l_dim,n_u=n_u,n_w=n_w,l_kernel=l_kernel,sig2_kernel=sig2_kernel)
        self.layer_1=nn.Linear(l_dim,20)
        self.layer_2=nn.Linear(20,20)
        self.layer_3=nn.Linear(20,1)
    def forward(self,idx_pat,idx_meas,idx_t):
        latent=((self.pat_lat(idx_pat)+torch.mm(self.covariates_u[idx_pat,:],self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(self.cov_w_fixed[idx_t,:],self.beta_w)))
        out=F.relu(self.layer_1(latent))
        out=F.relu(self.layer_2(out))
        out=F.relu(self.layer_3(out)).squeeze(1)
        return(out)

class By_pat_tensor_fact(tensor_fact):
    def __init__(self,device,covariates,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3,l_kernel=3,sig2_kernel=1):
        super(By_pat_tensor_fact,self).__init__(device=device,covariates=covariates,n_pat=n_pat,n_meas=n_meas,n_t=n_t,l_dim=l_dim,n_u=n_u,n_w=n_w,l_kernel=l_kernel,sig2_kernel=sig2_kernel)
    def forward(self,idx_pat):
        #cov_w=torch.tensor(range(0,101)).unsqueeze(1)#.double()
        pred=torch.einsum('il,jkl->ijk',((self.pat_lat(idx_pat)+torch.mm(self.covariates_u[idx_pat,:],self.beta_u),torch.einsum("il,jl->ijl",(self.meas_lat.weight,(self.time_lat.weight+torch.mm(self.cov_w_fixed,self.beta_w)))))))
        #pred=((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u))*(self.meas_lat.weight)*(self.time_lat.weight+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)



class TensorFactDataset(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor.csv",file_path="~/Data/MIMIC/",cov_path="complete_covariates",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.length=len(self.lab_short.index)
        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        self.meas_num=self.lab_short["LABEL_CODE"].nunique()

        self.time_values=["TIME_STAMP","TIME_SQ"]

        #Randomly select patients for classification validation.
        #self.test_idx=np.random.choice(self.pat_num,size=int(0.2*self.pat_num),replace=False) #0.2 validation rate
        #self.lab_short.loc[self.lab_short["UNIQUE_ID"].isin(self.test_idx),"DEATHTAG"]=np.nan
        self.tensor_mat=self.lab_short.as_matrix()

        #self.tags=pd.read_csv(file_path+"death_tag_tensor.csv")
        #self.test_labels=torch.tensor(self.tags.loc[self.tags["UNIQUE_ID"].isin(self.test_idx)].sort_values(by="UNIQUE_ID").as_matrix())
       # self.test_covariates=torch.tensor(self.lab_short.loc[self.lab_short["UNIQUE_ID"].isin(self.test_idx)].sort_values(by="UNIQUE_ID")[self.cov_values].as_matrix()).to(torch.double)
        #covariates=self.lab_short.groupby("UNIQUE_ID").first()[self.cov_values].reset_index()
        #self.test_covariates=torch.tensor(covariates.loc[covariates["UNIQUE_ID"].isin(self.test_idx)].sort_values(by="UNIQUE_ID")[self.cov_values].as_matrix()).to(torch.double)
        #print(self.test_covariates.size())
        #print(self.test_labels.size())

        self.cov_u=torch.tensor(pd.read_csv(file_path+cov_path+".csv").as_matrix()[:,1:]).to(torch.double)
        self.covu_num=self.cov_u.size(1)

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return(self.tensor_mat[idx,:])

class TensorFactDataset_ByPat(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor.csv",file_path="~/Data/MIMIC/",cov_path="complete_covariates",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        idx_mat=self.lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]].as_matrix()
        idx_tens=torch.LongTensor(idx_mat[:,:-1])
        val_tens=torch.DoubleTensor(idx_mat[:,-1])
        sparse_data=torch.sparse.DoubleTensor(idx_tens.t(),val_tens)
        self.data_matrix=sparse_data.to_dense()

        self.cov_u=torch.tensor(pd.read_csv(file_path+cov_path+".csv").as_matrix()[:,1:]).to(torch.double)
        self.length=self.cov_u.size(0)
        self.covu_num=self.cov_u.size(1)

        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        self.meas_num=self.lab_short["LABEL_CODE"].nunique()

        #Computations for the classification setting
        #self.tags=pd.read_csv(file_path+"death_tag_tensor.csv").as_matrix()[:,1]
        #self.test_idx=np.random.choice(self.length,size=int(0.2*self.length),replace=False) #0.2 validation rate
        #self.train_tags=self.tags
        #self.train_tags[self.test_idx]=np.nan

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return([idx,self.data_matrix[idx,:,:]])#,self.train_tags[idx]])

def mod_select(opt,tensor_path="complete_tensor",cov_path="complete_covariates"):

    N_t=97 # NUmber of time steps

    str_dir="./trained_models/"
    for key in vars(opt):
        str_dir+=str(key)+str(vars(opt)[key])+"_"
    str_dir+="/"

    #Check if output directory exits, otherwise, create it.
    if (not os.path.exists(str_dir)):
        os.makedirs(str_dir)
    else:
        replace_prev=input("This configuration has already been run !Do you want to continue ? y/n")
        if (replace_prev=="n"):
            raise ValueError("Aborted")
    #Gpu selection
    gpu_id="1"
    if opt.gpu_name=="Tesla":
        gpu_id="0"

    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

    if opt.cuda:
        device=torch.device("cuda:0")
    else:
        device=torch.device("cpu")
    print("Device : "+str(device))

    print("GPU num used : "+str(torch.cuda.current_device()))
    print("GPU used : "+str(torch.cuda.get_device_name(torch.cuda.current_device())))

    if opt.hard_split:
        suffix="_HARD.csv"
    else:
        suffix=".csv"

    if opt.by_pat:
        if opt.DL:
            raise ValueError("Deep Learning with data feeding by patient is not supported yet")
        elif opt.death_label:
            train_dataset=TensorFactDataset_ByPat(csv_file_serie="complete_tensor.csv") #Full dataset for the Training
        else:
            train_dataset=TensorFactDataset_ByPat(csv_file_serie=tensor_path+"_train"+suffix,cov_path=cov_path)
            val_dataset=TensorFactDataset_ByPat(csv_file_serie=tensor_path+"_val"+suffix,cov_path=cov_path)

        mod=By_pat_tensor_fact(device=device,covariates=train_dataset.cov_u,n_pat=train_dataset.length,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
        mod.double()
        mod.to(device)
    else:
        if opt.DL:
            train_dataset=TensorFactDataset(csv_file_serie=tensor_path+"_train"+suffix,cov_path=cov_path)
            val_dataset=TensorFactDataset(csv_file_serie=tensor_path+"_val"+suffix,cov_path=cov_path)
            mod=deep_tensor_fact(device=device,covariates=train_dataset.cov_u,n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
        elif opt.death_label:
            train_dataset=TensorFactDataset(csv_file_serie="complete_tensor.csv",cov_path=cov_path) #Full dataset for the Training
            mod=tensor_fact(device=device,covariates=train_dataset.cov_u,n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
        elif opt.XT:
            train_dataset=TensorFactDataset(csv_file_serie=tensor_path+"_train"+suffix,cov_path=cov_path)
            val_dataset=TensorFactDataset(csv_file_serie=tensor_path+"_val"+suffix,cov_path=cov_path)
            mod=XT_tensor_fact(device=device,covariates=train_dataset.cov_u,n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
        else:
            train_dataset=TensorFactDataset(csv_file_serie=tensor_path+"_train"+suffix,cov_path=cov_path)
            val_dataset=TensorFactDataset(csv_file_serie=tensor_path+"_val"+suffix,cov_path=cov_path)
            mod=tensor_fact(device=device,covariates=train_dataset.cov_u,n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)

    dataloader = DataLoader(train_dataset, batch_size=opt.batch,shuffle=True,num_workers=2)
    dataloader_val=None
    if not opt.death_label:
        dataloader_val = DataLoader(val_dataset, batch_size=len(val_dataset),shuffle=False)

    return(dataloader,dataloader_val,mod,device,str_dir)
