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

import random

from GRU_teach import GRU_teach, GRU_teach_dataset

data_train=GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/")
data_val=GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/",csv_file_serie="LSTM_tensor_val.csv",cov_path="LSTM_covariates_val.csv",tag_path="LSTM_death_tags_val.csv")
data_test=GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/",csv_file_serie="LSTM_tensor_test.csv",cov_path="LSTM_covariates_test.csv",tag_path="LSTM_death_tags_test.csv")

dataloader=DataLoader(data_train,batch_size=5000,shuffle=True)
device=torch.device("cuda:0")
mod_num=3 #number of models to train.
results_dict=dict()
config=dict()
for mod_idx in range(mod_num):
    name=f"model_{mod_idx}"
    config["L2"]=10**(3*random.random()-8)
    config["mixing_ratio"]=random.random()

    mod=GRU_teach(device,data_train.cov_u.size(1),data_train.data_matrix.size(1))
    mod.to(device)

    optimizer=torch.optim.Adam(self.mod.parameters(), lr=l_r, weight_decay=config["L2"])
    criterion=nn.MSELoss(reduce=False,size_average=False)
    class_criterion=nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        for i_batch,sampled_batch in enumerate(self.dataloader):
            optimizer.zero_grad()
            [preds,class_preds]=mod.forward(sampled_batch[2].to(device),sampled_batch[0].to(device),sampled_batch[1].to(device))
            targets=sampled_batch[0].to(device)
            targets.masked_fill_(1-sampled_batch[1].to(device),0)
            preds.masked_fill_(1-sampled_batch[1].to(device),0)
            loss=(1-config["mixing_ratio"])*(torch.sum(criterion(preds,targets))/torch.sum(sampled_batch[1].to(device)).float())+config["mixing_ratio"]*class_criterion(class_preds,sampled_batch[3].to(device))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            [preds,class_preds]=mod.forward(data_val.cov_u.to(device),data_val.data_matrix.to(device),data_val.observed_mask.to(device))
            targets=data_val.data_matrix.to(device)
            targets.masked_fill_(1-data_val.observed_mask.to(device),0)
            preds.masked_fill_(1-data_val.observed_mask.to(device),0)
            #loss_val=class_criterion(class_preds,get_pinned_object(data_val).tags.to(self.device)).cpu().detach().numpy()
            loss_val=roc_auc_score(data_val.tags,class_preds.cpu())
            print("Validation Loss")
            print(loss_val)
    results_dict["name"]=loss_val
    torch.save(mod.state_dict(),name+".pt")
