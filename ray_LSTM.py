

import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed",RuntimeWarning)
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed",RuntimeWarning)

import random
import numpy as np
import ray
import ray.tune as tune
from ray.tune.hyperband import HyperBandScheduler
#from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler
from ray.tune import Trainable, TrainingResult
from ray.tune.util import pin_in_object_store, get_pinned_object

import torch
import torch.nn as nn
from tensor_utils import TensorFactDataset, model_selector

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
from baselines import GRU_mod

import os
import shutil

class reconstructed_ts(Dataset):
    def __init__(self,latents_pat,latents_feat,latents_time,tags,real_data_path,sample):
        #True samples
        lab_short=pd.read_csv(real_data_path)
        idx_mat=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]].as_matrix()
        idx_tens=torch.LongTensor(idx_mat[:,:-1])
        val_tens=torch.DoubleTensor(idx_mat[:,-1])
        sparse_data=torch.sparse.DoubleTensor(idx_tens.t(),val_tens)
        data_matrix=sparse_data.to_dense()[sample,:,:]
        #Reconstructed data
        reconstructed_tensor=torch.Tensor(np.einsum('il,jl,kl->ijk',latents_pat,latents_feat,latents_times))

        mask_true=(data_matrix!=0) #is 1 if sample is present
        self.full_tensor=reconstructed_tensor.masked_scatter(mask_true,data_matrix)

        self.tags
    def __len__(self):
        return(self.full_tensor.size(0))
    def __getitem__(self,idx):
        return([self.full_tensor[idx,:],self.tags[idx]])
    def get_dim(self):
        return(self.full_tensor.size(1))

class train_class(Trainable):
    def _setup(self):
        self.device=torch.device("cuda:0")
        mod_opt={'type':"plain_fact",'cov':False,'latents':20}

        #data_train=TensorFactDataset(csv_file_serie="complete_tensor_train1.csv",cov_path="complete_covariates")

        self.mod=GRU_mean(get_pinned_object(data_train).get_dim())

        self.dataloader = DataLoader(get_pinned_object(data_train),batch_size=5000,shuffle=True,num_workers=2)
        self.dataloader_val= DataLoader(get_pinned_object(data_val),batch_size=1000,shuffle=False)
        #self.dataloader=DataLoader(data_train,batch_size=65000,shuffle=True,num_workers=2)
        self.timestep=0
        print("SETUUUUP")
    def _train(self):
        self.timestep+=1

        print("Timestep")
        print(self.timestep)

        #Select learning rate depending on the epoch.
        if self.timestep<40:
            l_r=0.005
        elif self.timestep<60:
            l_r=0.0015
        else:
            l_r=0.0005

        optimizer = torch.optim.Adam(self.mod.parameters(), lr=l_r, weight_decay=self.config["L2"])

        criterion=nn.BCELoss()
        total_loss=0
        for idx, sampled_batch in enumerate(self.dataloader):
            optimizer.zero_grad()
            optimizer_w.zero_grad()
            target=sampled_batch[1]
            preds=self.mod.fwd(sampled_batch[0])
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_val=0
            for i_val,batch_val in enumerate(self.dataloader_val):
                target=batch_val[1]
                preds=self.mod.fwd(batch_val[0])
                loss_val+=roc_auc_score(target,preds)
        auc_mean=loss_val/(i_val+1)
        #rmse_val_loss_computed=(np.sqrt(loss_val.detach().cpu().numpy()/(i_val+1)))

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

file_path_pat=sys.argv[1:][0] # This file should contain a numpy array with the latents and the label as first columnself.
file_path_feat=sys.argv[1:][1]
file_path_time=sys.argv[1:][2]

ray.init(num_cpus=10)


latents_pat=np.load(file_path_pat)
latents_feat=np.load(file_path_feat)
latents_time=np.load(file_path_time)

tags=pd.read_csv("~/Data/MIMIC/LSTM_death_tags_train.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

train_idx,test_idx=train_test_split(np.arange(tag_mat.shape[0]),test_size=0.2,random_state=42)

data_train=pin_in_object_store(reconstructed_ts(latents_pat[train_idx],latents_feat[train_idx],latents_time[train_idx],tag_mat[train_idx]),"~/Data/MIMIC/LSTM_tensor_train.csv",train_idx)
data_train=pin_in_object_store(reconstructed_ts(latents_pat[test_idx],latents_feat[test_idx],latents_time[test_idx],tag_mat[test_idx]),"~/Data/MIMIC/LSTM_tensor_train.csv",test_idx)


tune.register_trainable("my_class", train_class)

hyperband=HyperBandScheduler(time_attr="timesteps_total",reward_attr="mean_accuracy",max_t=100)

exp={
        'run':"my_class",
        'repeat':50,
        'stop':{"training_iteration":1},
        'config':{
        "L2":lambda spec: 10**(8*random.random()-4)
    }
 }

tune.run_experiments({"Classification_example":exp},scheduler=hyperband)
