import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed",RuntimeWarning)
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed",RuntimeWarning)

import random
import numpy as np
import pandas as pd
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

from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import os
import shutil
import sys

class train_class(Trainable):
    def _setup(self):
        self.timestep=0
        self.clf=svm.SVC(C=self.config["C"],class_weight="balanced",probability=True,kernel="linear",gamma=self.config["gamma"])
    def _train(self):
        self.timestep+=1

        n_splits=2
        skf = StratifiedKFold(n_splits=n_splits,random_state=7)
        auc=0
        for train_index, val_index in skf.split(get_pinned_object(x),get_pinned_object(y)):
            x_train=get_pinned_object(x)[train_index,:]
            y_train=get_pinned_object(y)[train_index]
            x_val=get_pinned_object(x)[val_index,:]
            y_val=get_pinned_object(y)[val_index]
            probas_=self.clf.fit(x_train,y_train).predict_proba(x_val)
            print(y_val.shape)
            print(probas_.shape)
            auc+=roc_auc_score(y_val,probas_[:,1])

        return TrainingResult(mean_accuracy=auc/n_splits,timesteps_this_iter=1)

    def _save(self,checkpoint_dir):
        return path
    def _restore(self,checkpoint_path):
        return checkpoint_path

file_path=sys.argv[1:][0] # This file should contain a numpy array with the latents and the label as first columnself.

ray.init(num_cpus=3)

latents=np.load(file_path)
tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]
x=pin_in_object_store(latents.T)
y=pin_in_object_store(tag_mat)

tune.register_trainable("my_class", train_class)

hyperband=HyperBandScheduler(time_attr="timesteps_total",reward_attr="mean_accuracy",max_t=100)

exp={
        'run':"my_class",
        'repeat':50,
        'stop':{"training_iteration":1},
        'config':{
        "C":lambda spec: 10**(8*random.random()-4),
        "gamma":lambda spec: 10**(8*random.random()-4),
    }
 }

tune.run_experiments({"Classification_example":exp},scheduler=hyperband)
