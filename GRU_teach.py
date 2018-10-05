import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

import ray
import ray.tune as tune
from ray.tune.schedulers import AsyncHyperBandScheduler
#from ray.tune.schedulers import AsyncHyperBandScheduler,HyperBandScheduler
from ray.tune import Trainable#, TrainingResult
from ray.tune.util import pin_in_object_store, get_pinned_object
import random

from hyperopt import hp
from ray.tune.suggest import HyperOptSearch

from GRU_utils import GRU_teach, GRU_teach_dataset

def train():
    data_train=GRU_teach_dataset()
    data_val=GRU_teach_dataset(csv_file_serie="LSTM_tensor_val.csv",cov_path="LSTM_covariates_val.csv")
    dataloader=DataLoader(data_train,batch_size=5000,shuffle=True,num_workers=2)
    device=torch.device("cpu")
    mod=GRU_teach(device,data_train.cov_u.size(1),data_train.data_matrix.size(1))
    mod.float()

    optimizer=torch.optim.Adam(mod.parameters(),lr=0.005,weight_decay=0.00005)
    criterion=nn.MSELoss(reduce=False,size_average=False)
    for epoch in range(100):
        for i_batch, sampled_batch in enumerate(dataloader):

            optimizer.zero_grad()

            [preds,class_pred]=mod.forward(sampled_batch[2],sampled_batch[0],sampled_batch[1])

            targets=sampled_batch[0]
            targets.masked_fill_(1-sampled_batch[1],0)
            preds.masked_fill_(1-sampled_batch[1],0)

            loss=torch.sum(criterion(preds,targets))/torch.sum(sampled_batch[1]).float()

            loss.backward()
            optimizer.step()
            print(loss.detach().numpy())

        with torch.no_grad():
            [preds,class_pred]=mod.forward(data_val.cov_u,data_val.data_matrix,data_val.observed_mask)
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

        self.mod=GRU_teach(self.device,get_pinned_object(data_train).cov_u.size(1),get_pinned_object(data_train).data_matrix.size(1))
        self.mod.float()
        self.mod.to(self.device)

        self.dataloader=DataLoader(get_pinned_object(data_train),batch_size=5000,shuffle=True,num_workers=2)

        self.timestep=0
    def _train(self):
        self.timestep+=1

        #Select learning rate depending on the epoch.
        if self.timestep<50:
            l_r=0.0005
        elif self.timestep<100:
            l_r=0.00015
        elif self.timestep<150:
            l_r=0.00005
        else:
            l_r=0.00001

        optimizer = torch.optim.Adam(self.mod.parameters(), lr=l_r, weight_decay=self.config["L2"])

        criterion=nn.MSELoss(reduce=False,size_average=False)
        class_criterion=nn.BCELoss()

        for i_batch,sampled_batch in enumerate(self.dataloader):
            optimizer.zero_grad()
            [preds,class_preds]=self.mod.forward(sampled_batch[2].to(self.device),sampled_batch[0].to(self.device),sampled_batch[1].to(self.device))
            targets=sampled_batch[0].to(self.device)
            targets.masked_fill_(1-sampled_batch[1].to(self.device),0)
            preds.masked_fill_(1-sampled_batch[1].to(self.device),0)
            loss=(1-self.config["mixing_ratio"])*(torch.sum(criterion(preds,targets))/torch.sum(sampled_batch[1].to(self.device)).float())+self.config["mixing_ratio"]*class_criterion(class_preds,sampled_batch[3].to(self.device))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            [preds,class_preds]=self.mod.forward(get_pinned_object(data_val).cov_u.to(self.device),get_pinned_object(data_val).data_matrix.to(self.device),get_pinned_object(data_val).observed_mask.to(self.device))
            targets=get_pinned_object(data_val).data_matrix.to(self.device)
            targets.masked_fill_(1-get_pinned_object(data_val).observed_mask.to(self.device),0)
            preds.masked_fill_(1-get_pinned_object(data_val).observed_mask.to(self.device),0)
            #loss_val=class_criterion(class_preds,get_pinned_object(data_val).tags.to(self.device)).cpu().detach().numpy()
            loss_val=roc_auc_score(get_pinned_object(data_val).tags,class_preds.cpu())
            print("Validation Loss")
            print(loss_val)

            [preds,class_preds]=self.mod.forward(get_pinned_object(data_test).cov_u.to(self.device),get_pinned_object(data_test).data_matrix.to(self.device),get_pinned_object(data_test).observed_mask.to(self.device))
            targets=get_pinned_object(data_test).data_matrix.to(self.device)
            targets.masked_fill_(1-get_pinned_object(data_test).observed_mask.to(self.device),0)
            preds.masked_fill_(1-get_pinned_object(data_test).observed_mask.to(self.device),0)
            #loss_val=class_criterion(class_preds,get_pinned_object(data_val).tags.to(self.device)).cpu().detach().numpy()
            loss_test=roc_auc_score(get_pinned_object(data_test).tags,class_preds.cpu())

        return {'mean_accuracy':loss_val,'timesteps_this_iter':1,'test_auc':loss_test}

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
    data_train=pin_in_object_store(GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/"))
    data_val=pin_in_object_store(GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/",csv_file_serie="LSTM_tensor_val.csv",cov_path="LSTM_covariates_val.csv",tag_path="LSTM_death_tags_val.csv"))
    data_test=pin_in_object_store(GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/",csv_file_serie="LSTM_tensor_test.csv",cov_path="LSTM_covariates_test.csv",tag_path="LSTM_death_tags_test.csv"))

    tune.register_trainable("my_class", train_class)

    hyperband=AsyncHyperBandScheduler(time_attr="training_iteration",reward_attr="mean_accuracy",max_t=200,grace_period=15)

    space= {
            'L2':hp.loguniform('L2',-2.3*5,-2.3*9),
            'mixing_ratio':hp.uniform('mixing_ratio',0.9,1)
            }
    exp={
            'run':"my_class",
            'num_samples':50,
            'stop':{"training_iteration":200},
            'trial_resources':{
                            "gpu":1,
                            "cpu":1},
            'config':{
            "L2":lambda spec: 10**(3*random.random()-8),
            "mixing_ratio":lambda spec : random.random()
        }
     }
    algo=HyperOptSearch(space,reward_attr="mean_accuracy")

    tune.run_experiments({"GRU_teach":exp},search_alg=algo,scheduler=hyperband)

    print("Finished with the simulations")
