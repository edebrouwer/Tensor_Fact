#LSTM Classifier on the latent/observed time series.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Sequence(nn.Module):
    def __init__(self,input_dim,device):
        print("Model Initialization")
        super(Sequence, self).__init__()
        self.device=device
        self.input_dim=input_dim
        self.lstm_layers=2
        self.lstm1 = nn.LSTM(input_dim, 20, num_layers=self.lstm_layers)
        #self.lstm2 = nn.LSTM(20,20)
        self.class_layer1=nn.Linear(20,20)
        self.class_layer2=nn.Linear(20,1)

    def fwd(self,data_in,batch_dim):

        h_t = torch.zeros(self.lstm_layers,batch_dim, 20,device=self.device, dtype=torch.double)
        c_t = torch.zeros(self.lstm_layers,batch_dim, 20,device=self.device, dtype=torch.double)
        #Data should be in the lengthxbatchxdim
        _,(h_n,c_n)=self.lstm1(data_in,(h_t,c_t))


        out=F.relu(self.class_layer1(h_n))
        out=F.sigmoid(self.class_layer2(out))

        return out

def dummy_data_gen(time_steps,batch_size,in_size):
    dat=torch.zeros(batch_size,time_steps,in_size,dtype=torch.double)
    label=torch.zeros(batch_size,dtype=torch.double)
    for i in range(batch_size):
        label[i]=int(np.random.binomial(1,0.5))
        for j in range(in_size):
            dat[i,:,j]=torch.sin((1+label[i])*torch.linspace(0,6,time_steps,dtype=torch.double))
    return(dat,label)

def ford_data_load(Extension):
    #Extension is either FordB_TRAIN or FordB_TEST
    import pandas as pd
    df=pd.read_csv("~/Data/FordA/"+Extension+".txt",header=None)
    df.rename(columns={df.columns[0]: "Label" },inplace=True)
    df_np=df.values
    label=(torch.tensor(df_np[:,0])+1)/2
    dat=torch.tensor(df_np[:,1:]).unsqueeze(2)
    return(dat,label)

def earthquake_data_load(Extension):
    #Extension is either FordB_TRAIN or FordB_TEST
    import pandas as pd
    df=pd.read_csv("~/Data/Earthquakes/"+Extension+".txt",header=None)
    df.rename(columns={df.columns[0]: "Label" },inplace=True)
    df_np=df.values
    label=torch.tensor(df_np[:,0])
    dat=torch.tensor(df_np[:,1:]).unsqueeze(2)
    return(dat,label)

class series_dataset(Dataset):
    #Data is returned in the format batchxlengthxdim
    def __init__(self,file_path=None,data_source=None,time_steps=None,batch_size=None,in_size=None):
        if data_source=="dummy":
            self.series,self.labels=dummy_data_gen(time_steps,batch_size,in_size)
        elif "Ford" in data_source:
            self.series,self.labels=ford_data_load(Extension=data_source)
        elif "Earthquake" in data_source:
            self.series,self.labels=earthquake_data_load(Extension=data_source)
        elif "macau" in file_path:
            latent_pat=np.load(file_path+"mean_pat_latent.npy").T
            raise ValueError("Implementation not finished")
        else:
            raise ValueError("Data loading option not supported")
    def __len__(self):
        return self.series.size(0)
    def __getitem__(self,idx):
        return([self.series[idx,:,:],self.labels[idx]])


def train_model(model,dataloader,dataloader_val,device):
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01)#,weight_decay=0.002)
    criterion = nn.BCELoss()#
    epochs_num=100

    for epoch in range(epochs_num):
        total_loss=0
        for i_batch,sampled_batch in enumerate(dataloader):
            optimizer.zero_grad()
            pred=model.fwd(torch.transpose(sampled_batch[0],1,0).to(device),len(sampled_batch[0]))[0,:,0]
            loss=criterion(pred,sampled_batch[1].to(device))
            loss.backward()
            optimizer.step()
            total_loss+=loss
        with torch.no_grad():
            for i_batch_val,sampled_batch in enumerate(dataloader_val):
                val_pred=model.fwd(torch.transpose(sampled_batch[0],1,0).to(device),len(sampled_batch[0]))[0,:,0]
                val_loss=criterion(val_pred,sampled_batch[1].to(device))
        print("Loss for epoch {} = {} with val loss = {}".format(epoch,total_loss/(i_batch+1),val_loss/(i_batch_val+1)))
    return(model)

def run_dummy_experiment(batch_size,time_steps,in_size):
    device=torch.device("cpu")
    dummy_data=series_dataset(data_source="dummy",time_steps=time_steps,batch_size=batch_size,in_size=in_size)
    dummy_data_val=series_dataset(data_source="dummy",time_steps=time_steps,batch_size=batch_size,in_size=in_size)
    dataloader= DataLoader(dummy_data, batch_size=len(dummy_data),shuffle=True,num_workers=2)
    dataloader_val= DataLoader(dummy_data_val, batch_size=len(dummy_data),shuffle=True,num_workers=2)
    mod=Sequence(in_size,device)
    mod.double()
    train_model(mod,dataloader,dataloader_val,device)
    return(mod)

def run_ford_experiment():
    device=torch.device("cuda:0")
    ford_data=series_dataset(data_source="FordA_TRAIN")
    ford_data_test=series_dataset(data_source="FordA_TEST")
    dataloader= DataLoader(ford_data, batch_size=len(ford_data),shuffle=True,num_workers=2)
    dataloader_val= DataLoader(ford_data_test, batch_size=len(ford_data),shuffle=True,num_workers=2)
    mod=Sequence(1,device)
    mod.double()
    mod.to(device)
    train_model(mod,dataloader,dataloader_val,device)
    return(mod)

def run_earthquake_experiment():
    device=torch.device("cuda:0")
    earthquake_data=series_dataset(data_source="Earthquakes_TRAIN")
    earthquake_data_test=series_dataset(data_source="Earthquakes_TEST")
    dataloader= DataLoader(earthquake_data, batch_size=len(earthquake_data),shuffle=True,num_workers=2)
    dataloader_val= DataLoader(earthquake_data_test, batch_size=len(earthquake_data),shuffle=True,num_workers=2)
    mod=Sequence(1,device)
    mod.double()
    mod.to(device)
    train_model(mod,dataloader,dataloader_val,device)
    return(mod)


if __name__=="__main__":
    #mod=run_dummy_experiment(batch_size=50,time_steps=100,in_size=3)
    #mod=run_ford_experiment()
    mod=run_earthquake_experiment()
