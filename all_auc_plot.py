#Full AUC curve
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from MLP_class import MLP_class_mod, latent_dataset

from baselines import GRU_mean

from sklearn.metrics import roc_auc_score


means_df=pd.Series.from_csv("~/Data/MIMIC/mean_features.csv")
means_vec=torch.tensor(means_df.as_matrix(),dtype=tensor.float)
data_val=LSTMDataset_ByPat(csv_file_serie="LSTM_tensor_val.csv",file_path="~/Data/MIMIC/",cov_path="LSTM_covariates_val",tag_path="LSTM_death_tags_val.csv")
dataloader_val= DataLoader(data_val,batch_size=1000,shuffle=False)

#Mean model
device=torch.device("cuda:0")
mod=GRU_mean(data_train.meas_num,means_vec,device,imputation_mode="mean")

mod.load_state_dict(torch.load("GRU_mean_unique_model.pt"))

fpr,tpr,_ = roc_curve(mod.forward(torch.transpose(data_val.data_matrix.to(device),1,2)).cpu().detach().numpy(),data_val.tags.numpy())
roc_auc=auc(fpr,tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve Mean imputation (area = %0.2f)' % roc_auc)

#Simple model
mod=GRU_mean(data_train.meas_num,means_vec,device,imputation_mode="simple")
mod.load_state_dict(torch.load("GRU_simple_unique_model.pt"))
fpr,tpr,_ = roc_curve(mod.forward(torch.transpose(data_val.data_matrix.to(device),1,2)).cpu().detach().numpy(),data_val.tags.numpy())

roc_auc=auc(fpr,tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC curve Simple imputation (area = %0.2f)' % roc_auc)

#Macau Model
latents=np.load("results_macau_70/pca_latents.npy")
tags=pd.read_csv("~/Data/MIMIC/LSTM_death_tags_train.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

latents_train, latents_val, tag_mat_train, tag_mat_val=train_test_split(latents,tag_mat,test_size=0.2,random_state=42)

data_train=latent_dataset(latents_train,tag_mat_val)
data_val=latent_dataset(latents_val,tag_mat_val)

mod=MLP_class_mod(data_train.get_dim())

dataloader = DataLoader(data_train,batch_size=5000,shuffle=True,num_workers=2)
dataloader_val= DataLoader(data_val,batch_size=1000,shuffle=False)

for epoch in range(60):

    if self.timestep<40:
        l_r=0.005
    elif self.timestep<60:
        l_r=0.0015
    else:
        l_r=0.0005

    optimizer = torch.optim.Adam(mod.parameters(), lr=l_r, weight_decay=0.034084)

    criterion=nn.BCELoss()

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

    print(auc_mean.detach().numpy())
with torch.no_grad():
    fpr,tpr,_ = roc_curve(data_val.tags,mod.fwd(data_val.latents).detach().numpy())
    roc_auc=auc(fpr,tpr)

    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC Macau curve (area = %0.2f)' % roc_auc)




plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("Full_AUC.pdf")
