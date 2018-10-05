import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed",RuntimeWarning)
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed",RuntimeWarning)

import random
import numpy as np
import ray
import ray.tune as tune
#from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler
from ray.tune import Trainable#, TrainingResult
from ray.tune.util import pin_in_object_store, get_pinned_object

import torch
import torch.nn as nn
from tensor_utils import TensorFactDataset, model_selector

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score

from MLP_class import MLP_class_mod, latent_dataset

import os
import shutil
import sys
import pandas as pd

class train_class(Trainable):
    def _setup(self):
        self.device=torch.device("cuda:0")
        #mod_opt={'type':"plain_fact",'cov':False,'latents':20}

        #data_train=TensorFactDataset(csv_file_serie="complete_tensor_train1.csv",cov_path="complete_covariates")

        self.mod=MLP_class_mod(get_pinned_object(data_train).get_dim())
        self.mod.float()

        self.dataloader = DataLoader(get_pinned_object(data_train),batch_size=5000,shuffle=True,num_workers=2)
        self.dataloader_val= DataLoader(get_pinned_object(data_val),batch_size=len(get_pinned_object(data_val)),shuffle=False)
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

            target=sampled_batch[1].float()
            preds=self.mod.fwd(sampled_batch[0].float())
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_val=0
            for i_val,batch_val in enumerate(self.dataloader_val):
                target=batch_val[1].float()
                preds=self.mod.fwd(batch_val[0].float())
                loss_val+=roc_auc_score(target,preds)
        auc_mean=loss_val/(i_val+1)
        #rmse_val_loss_computed=(np.sqrt(loss_val.detach().cpu().numpy()/(i_val+1)))

        return {"mean_accuracy":auc_mean,"timesteps_this_iter":1}

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

file_path=sys.argv[1:][0] # This file should contain a numpy array with the latents and the label as first columnself.

ray.init(num_cpus=10)



latents=np.load(file_path)
n_train=pd.read_csv("~/Data/MIMIC/LSTM_tensor_train.csv")["UNIQUE_ID"].nunique()
latents_train=latents[:n_train,:]
latents_val=latents[n_train:,:]


tags_train=pd.read_csv("~/Data/MIMIC/LSTM_death_tags_train.csv").sort_values("UNIQUE_ID")
tag_mat_train=tags_train[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

tags_val=pd.read_csv("~/Data/MIMIC/LSTM_death_tags_val.csv").sort_values("UNIQUE_ID")
tag_mat_val=tags_val[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

data_train=pin_in_object_store(latent_dataset(latents_train,tag_mat_train))
data_val=pin_in_object_store(latent_dataset(latents_val,tag_mat_val))

tune.register_trainable("my_class", train_class)

hyperband=AsyncHyperBandScheduler(time_attr="training_iteration",reward_attr="mean_accuracy",max_t=100)

exp={
        'run':"my_class",
        'repeat':50,
        'stop':{"training_iteration":100},
        'config':{
        "L2":lambda spec: 10**(5*random.random()-6)
    }
 }

tune.run_experiments({"logistic_pca_50samples":exp},scheduler=hyperband)
