#Full AUC curve
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from MLP_class import MLP_class_mod, latent_dataset

from baselines import GRU_mean,LSTMDataset_ByPat

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import scipy.interpolate as interpolate

from GRU_utils import GRU_teach_dataset, GRU_teach

means_df=pd.Series.from_csv("~/Data/MIMIC/mean_features.csv")
means_vec=torch.tensor(means_df.as_matrix(),dtype=torch.float)
data_val=LSTMDataset_ByPat(csv_file_serie="LSTM_tensor_val.csv",file_path="~/Data/MIMIC/",cov_path="LSTM_covariates_val",tag_path="LSTM_death_tags_val.csv")
#dataloader_val= DataLoader(data_val,batch_size=1000,shuffle=False)

#Mean model
device=torch.device("cuda:0")
mod=GRU_mean(data_val.meas_num,means_vec,device,imputation_mode="mean")
mod.float()
mod.to(device)
mod.load_state_dict(torch.load("GRU_mean_unique_model.pt"))


fpr,tpr,_ = roc_curve(data_val.tags,mod.forward(torch.transpose(data_val.data_matrix.to(device),1,2)).cpu().detach().numpy())
np.save("./plots/fpr_Mean.npy",fpr)
np.save("./plots/tpr_Mean.npy",tpr)
roc_auc=auc(fpr,tpr)
plt.figure()
lw = 1
plt.plot(fpr, tpr, color='orange',lw=lw, label='ROC curve Mean imputation (area = %0.2f)' % roc_auc)

#Simple model
mod=GRU_mean(data_val.meas_num,means_vec,device,imputation_mode="simple")
mod.float()
mod.to(device)
mod.load_state_dict(torch.load("GRU_simple_unique_model.pt"))
fpr,tpr,_ = roc_curve(data_val.tags,mod.forward(torch.transpose(data_val.data_matrix.to(device),1,2)).cpu().detach().numpy())
np.save("./plots/fpr_Simple.npy",fpr)
np.save("./plots/tpr_Simple.npy",tpr)

roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr, color='royalblue',
lw=lw, label='ROC curve Simple imputation (area = %0.2f)' % roc_auc)

#Macau Model
latents=np.load("results_macau_70/pca_latents.npy")
n_train=pd.read_csv("~/Data/MIMIC/Clean_data/LSTM_tensor_train.csv")["UNIQUE_ID"].nunique()
n_val=pd.read_csv("~/Data/MIMIC/Clean_data/LSTM_tensor_val.csv")["UNIQUE_ID"].nunique()
latents_train=latents[:n_train,:]
latents_val=latents[n_train:n_train+n_val,:]

tags_train=pd.read_csv("~/Data/MIMIC/Clean_data/LSTM_death_tags_train.csv").sort_values("UNIQUE_ID")
tag_mat_train=tags_train[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

tags_val=pd.read_csv("~/Data/MIMIC/Clean_data/LSTM_death_tags_val.csv").sort_values("UNIQUE_ID")
tag_mat_val=tags_val[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

data_train=latent_dataset(latents_train,tag_mat_train)
data_val=latent_dataset(latents_val,tag_mat_val)

mod=MLP_class_mod(data_train.get_dim())

dataloader = DataLoader(data_train,batch_size=5000,shuffle=True,num_workers=2)
dataloader_val= DataLoader(data_val,batch_size=n_val,shuffle=False)

print(latents.shape)
print(n_train)
print(tag_mat_train.shape)
print(tag_mat_val.shape)


print(data_val.tags.shape)
print(data_val.latents.shape)

for epoch in range(60):

    if epoch<30:
        l_r=0.005
    elif epoch<40:
        l_r=0.0005
    else:
        l_r=0.0001

    optimizer = torch.optim.Adam(mod.parameters(), lr=l_r, weight_decay=0.034084)

    criterion=nn.BCELoss()

    for idx, sampled_batch in enumerate(dataloader):
        optimizer.zero_grad()
        #optimizer_w.zero_grad()
        target=sampled_batch[1].float()
        preds=mod.fwd(sampled_batch[0].float())
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        loss_val=0
        for i_val,batch_val in enumerate(dataloader_val):
            target=batch_val[1].float()
            preds=mod.fwd(batch_val[0].float())
            loss_val+=roc_auc_score(target,preds)
    auc_mean=loss_val/(i_val+1)
    #rmse_val_loss_computed=(np.sqrt(loss_val.detach().cpu().numpy()/(i_val+1)))

    print(auc_mean)

with torch.no_grad():

    fpr,tpr,_ = roc_curve(data_val.tags,mod.fwd(torch.tensor(data_val.latents).float()).detach().numpy())
    np.save("./plots/fpr_Macau.npy",fpr)
    np.save("./plots/tpr_Macau.npy",tpr)
    roc_auc=auc(fpr,tpr)

    plt.plot(fpr, tpr, color='green',
    lw=lw, label='ROC Macau curve (area = %0.2f)' % roc_auc)

#GRU_teach model.
data_train=GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/")
data_val=GRU_teach_dataset(file_path="~/Data/MIMIC/Clean_data/",csv_file_serie="LSTM_tensor_val.csv",cov_path="LSTM_covariates_val.csv",tag_path="LSTM_death_tags_val.csv")

L2=5.98239163e-8
mixing_ratio=0.92541

device=torch.device("cuda:0")
mod=GRU_teach(device,data_train.cov_u.size(1),data_train.data_matrix.size(1))
mod.float()
mod.to(device)

state_dict=torch.load("./poor_bayes_models/model_132.pt")
mod.load_state_dict(state_dict)



with torch.no_grad():
    [preds,class_preds]=mod.forward(data_val.cov_u.to(device),data_val.data_matrix.to(device),data_val.observed_mask.to(device))
    targets=data_val.data_matrix.to(device)
    #loss_val=class_criterion(class_preds,get_pinned_object(data_val).tags.to(self.device)).cpu().detach().numpy()
    fpr,tpr,_ = roc_curve(data_val.tags,class_preds.cpu())
    np.save("./plots/fpr_GRU.npy",fpr)
    np.save("./plots/tpr_GRU.npy",tpr)
    roc_auc=auc(fpr,tpr)
    print(roc_auc)



print("Start plots")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves of the different methods')
plt.legend(loc="lower right")
plt.savefig("Full_AUC.pdf")
