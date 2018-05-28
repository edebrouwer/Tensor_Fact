
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#Check Ethan Rosenthal github

class tensor_fact(nn.Module):
    def __init__(self,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3):
        super(tensor_fact,self).__init__()
        self.n_pat=n_pat
        self.n_meas=n_meas
        self.n_t=n_t
        self.l_dim=l_dim
        self.n_u=n_u
        self.n_w=n_w
        self.pat_lat=nn.Embedding(n_pat,l_dim) #sparse gradients ?
        self.pat_lat.weight=nn.Parameter(0.25*torch.randn([n_pat,l_dim]))
        self.meas_lat=nn.Embedding(n_meas,l_dim)
        self.meas_lat.weight=nn.Parameter(0.25*torch.randn([n_meas,l_dim]))
        self.time_lat=nn.Embedding(n_t,l_dim).double()
        self.beta_u=nn.Parameter(torch.randn([n_u,l_dim],requires_grad=True).double())
        self.beta_w=nn.Parameter(torch.randn([n_w,l_dim],requires_grad=True).double())
    def forward(self,idx_pat,idx_meas,idx_t,cov_u,cov_w):
        pred=((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)
    def forward_full(self,idx_pat,cov_u):
        cov_w=torch.tensor(range(0,101)).unsqueeze(1).double()
        pred=torch.einsum('il,jkl->ijk',((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u),torch.einsum("il,jl->ijl",(self.meas_lat.weight,(self.time_lat.weight+torch.mm(cov_w,self.beta_w)))))))
        #pred=((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u))*(self.meas_lat.weight)*(self.time_lat.weight+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)


class TensorFactDataset(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor.csv",file_path="~/Data/MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.length=len(self.lab_short.index)
        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        self.cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
        self.time_values=["TIME_STAMP","TIME_SQ"]
        self.tensor_mat=self.lab_short.as_matrix()
        #print(self.lab_short.dtypes)
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        #print(self.lab_short["VALUENUM"].iloc[idx].values)
        #return([torch.from_numpy(self.lab_short.iloc[idx][["UNIQUE_ID","LABEL_CODE","TIME_STAMP"]].astype('int64').as_matrix()),torch.from_numpy(self.lab_short.iloc[idx][self.cov_values].as_matrix()),torch.from_numpy(self.lab_short.iloc[idx][self.time_values].astype('float64').as_matrix()),torch.tensor(self.lab_short["VALUENUM"].iloc[idx],dtype=torch.double)])
        #return(self.lab_short.iloc[idx].as_matrix())
        return(self.tensor_mat[idx,:])

def main():
    #With Adam optimizer

    import time

    train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train.csv")
    val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val.csv")
    dataloader = DataLoader(train_dataset, batch_size=65000,shuffle=True,num_workers=30)
    dataloader_val = DataLoader(val_dataset, batch_size=len(val_dataset),shuffle=False)

    train_hist=np.array([])
    val_hist=np.array([])

    mod=tensor_fact(n_pat=train_dataset.pat_num,n_meas=30,n_t=101,l_dim=1,n_u=18,n_w=1)
    mod.double()

    optimizer=torch.optim.Adam(mod.parameters(), lr=0.01) #previously lr 0.03 with good rmse
    criterion = nn.MSELoss()#
    epochs_num=150

    for epoch in range(epochs_num):
        print("EPOCH : "+str(epoch))
        total_loss=0
        t_tot=0
        Epoch_time=time.time()
        for i_batch,sampled_batch in enumerate(dataloader):

            #print(i_batch)
            starttime=time.time()

            indexes=sampled_batch[:,1:4].to(torch.long)
            cov_u=sampled_batch[:,4:22]
            cov_w=sampled_batch[:,3].unsqueeze(1)
            target=sampled_batch[:,-1]

            optimizer.zero_grad()
            preds=mod.forward(indexes[:,0],indexes[:,1],indexes[:,2],cov_u,cov_w)
            loss=criterion(preds,target)
            loss.backward()
            optimizer.step()
            t_flag=time.time()-starttime
            #print(t_flag)
            t_tot+=t_flag
            total_loss+=loss
        print("Current Training Loss :"+str(total_loss/(i_batch+1)))
        print("TOTAL TIME :"+str(time.time()-Epoch_time))
        print("Computation Time:"+str(t_tot))
        train_hist=np.append(train_hist,total_loss.item()/(i_batch+1))

        with torch.no_grad():
            for i_val,batch_val in enumerate(dataloader_val):
                indexes=batch_val[:,1:4].to(torch.long)
                cov_u=batch_val[:,4:22]
                cov_w=batch_val[:,3].unsqueeze(1)
                target=batch_val[:,-1]
                pred_val=mod.forward(indexes[:,0],indexes[:,1],indexes[:,2],cov_u,cov_w)
                loss_val=criterion(pred_val,target)
                print("Validation Loss :"+str(loss_val))
                val_hist=np.append(val_hist,loss_val)

    torch.save(mod.state_dict(),"current_model.pt")
    torch.save(train_hist,"train_history.pt")
    torch.save(val_hist,"validation_history.pt")

if __name__=="__main__":
    main()
