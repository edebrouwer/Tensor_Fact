import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed",RuntimeWarning)
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed",RuntimeWarning)

import random
import numpy as np
import ray
import ray.tune as tune
from ray.tune.schedulers import HyperBandScheduler
#from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler
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
from PCA_samples import PCA_macau_samples
import os
import shutil
import sys


if __name__=="__main__":


    file_path=sys.argv[1:][0] # This file should contain a numpy array with the latents and the label as first columnself.


    latents=np.load(file_path)
    tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")

    run_ray_logistic(file_path,tags,None,None)



def run_ray_logistic(latents_path,tags,kf,idx,log_name):

    ray.init(num_cpus=5,num_gpus=1)
    data_train_list=[]
    data_val_list=[]
    for train_idx, val_idx in kf.split(idx):
        train_idx=idx[train_idx] #Indexes from the full tensor.
        val_idx=idx[val_idx] #Indexes from the full tensor.

        latents_train, latents_val=PCA_macau_samples(dir_path=latents_path,idx_train=train_idx,idx_val=val_idx)

        data_train_list+=[latent_dataset(latents_train,tags[train_idx])]
        data_val_list+=[latent_dataset(latents_val,tags[val_idx])]

    data_train=pin_in_object_store(data_train_list)
    data_val=pin_in_object_store(data_val_list)


    class train_class(Trainable):
        def _setup(self):
            self.device=torch.device("cuda:0")
            mod_opt={'type':"plain_fact",'cov':False,'latents':20}
            self.nfolds=self.config["nfolds"]
            #data_train=TensorFactDataset(csv_file_serie="complete_tensor_train1.csv",cov_path="complete_covariates")
            self.mod=[]
            self.dataloader=[]
            self.data_val=get_pinned_object(data_val)
            for fold in range(self.nfolds):
                mod_fold=MLP_class_mod(get_pinned_object(data_train)[fold].get_dim())
                mod_fold.to(self.device)
                self.mod+=[mod_fold]
            #self.mod=MLP_class_mod(get_pinned_object(data_train).get_dim())

                self.dataloader += [DataLoader(get_pinned_object(data_train)[fold],batch_size=5000,shuffle=True)]
                #self.dataloader_val += DataLoader(get_pinned_object(data_val),batch_size=1000,shuffle=False)
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

            auc_mean_folds=0
            for fold in range(self.nfolds):
                optimizer = torch.optim.Adam(self.mod[fold].parameters(), lr=l_r, weight_decay=self.config["L2"])

                criterion=nn.BCEWithLogitsLoss()
                total_loss=0
                for idx, sampled_batch in enumerate(self.dataloader[fold]):
                    optimizer.zero_grad()
                    target=sampled_batch[1].to(self.device)
                    preds=self.mod[fold].fwd(sampled_batch[0].to(self.device))
                    loss = criterion(preds, target)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    loss_val=0
                    target=self.data_val[fold].tags.to(self.device)
                    preds=self.mod[fold].fwd(self.data_val[fold].latents.to(self.device))
                    loss_val+=roc_auc_score(target,preds)
                    auc_mean=loss_val
                #rmse_val_loss_computed=(np.sqrt(loss_val.detach().cpu().numpy()/(i_val+1)))
                auc_mean_folds+=auc_mean

            #return TrainingResult(mean_accuracy=(auc_mean_folds/self.nfolds),timesteps_this_iter=1)
            return {"mean_accuracy":(auc_mean_folds/self.nfolds),"time_steps_this_iter":1}
        def _save(self,checkpoint_dir):
            print("Saving")
            path=os.path.join(checkpoint_dir,"checkpoint")
            state_dict_list=[]
            for fold in range(self.nfolds):
                state_dict_list+=[self.mod[fold].state_dict()]
            torch.save(state_dict_list,path)
            print("SAVIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIING")
            #raise Exception()
            #torch.cuda.empty_cache()
            np.save(path+"_timestep.npy",self.timestep)
            return path
        def _restore(self,checkpoint_path):
            print("LOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADING")
            state_dict_list=torch.load(checkpoint_path)
            for fold in range(self.nfolds):
                self.mod[fold].load_state_dict(state_dict_list[fold])
            self.timestep=np.load(checkpoint_path+"_timestep.npy").item()



    tune.register_trainable("my_class", train_class)

    hyperband=HyperBandScheduler(time_attr="timesteps_total",reward_attr="mean_accuracy",max_t=100)

    exp={
            'run':"my_class",
            'num_samples':50,
            'trial_resources':{"gpu":1},
            'stop':{"training_iteration":100},
            'config':{
            "L2":lambda spec: 10**(8*random.random()-4),
            "nfolds":kf.get_n_splits()
        }
     }

    tune.run_experiments({log_name:exp},scheduler=hyperband)
