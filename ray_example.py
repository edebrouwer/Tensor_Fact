
import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed",RuntimeWarning)
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed",RuntimeWarning)

import random
import numpy as np
import ray 
import ray.tune as tune
from ray.tune.hyperband import HyperBandScheduler
from ray.tune import Trainable

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os

class model(nn.Module):

    def __init__(self):
        super(model,self).__init__()
        self.layer1=nn.Linear(1,5)
        self.layer2=nn.Linear(5,1)
    def forward(self,x):
        out=F.relu(self.layer1(x.unsqueeze(1)))
        out=self.layer2(out)
        return(out)

def train_func(config, reporter):  # add a reporter arg
    mod = model()
    criterion=nn.MSELoss()
    optimizer = torch.optim.Adam(mod.parameters(), lr=config["lr"])
    dataloader = DataLoader(mock_dataset(),batch_size=1000,shuffle=True,num_workers=2)
    total_loss=0
    for idx, (data, target) in enumerate(dataloader):
        
        optimizer.zero_grad()
        output = mod.forward(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss+=loss
    reporter(timesteps_total=idx,mean_loss=loss) # report metrics

class mock_dataset(Dataset):
    def __init__(self):
        self.x=10*torch.randn(10000)
        #self.y=(self.x>0).float()
        self.y=torch.sin(self.x)+0.01*torch.randn(10000)
    def __len__(self):
        return(self.x.size(0))
    def __getitem__(self,idx):
        return([self.x[idx],self.y[idx]])


class train_class(Trainable):
    def _setup(self):
        self.device=torch.device("cuda:0")
        self.mod=model()
        self.mod.to(self.device)
        self.timestep=0
    def _train(self):
        self.timestep+=1
        optimizer = torch.optim.Adam(self.mod.parameters(), lr=self.config["lr"], weight_decay=self.config["L2"])  
        criterion=nn.MSELoss()
        total_loss=0
        dataloader = DataLoader(mock_dataset(),batch_size=1000,shuffle=True,num_workers=2)
        for idx, (data, target) in enumerate(dataloader):    
            optimizer.zero_grad()
            output = self.mod.forward(data.to(self.device))
            loss = criterion(output, target.unsqueeze(1).to(self.device))
            loss.backward()
            optimizer.step()
            total_loss+=loss
        mean_loss=(total_loss.detach().cpu().numpy()/(idx+1))
        return {"mean_loss":mean_loss,"training_iteration":self.timestep}
    def _save(self,checkpoint_dir):
        path=os.path.join(checkpoint_dir,"checkpoint.pt")
        torch.save(self.mod.state_dict(),path)
        return path
    def _restore(self,checkpoint_path):
        self.mod.load_state_dict(torch.load(checkpoint_path))

#train_func(config={"lr":0},reporter=0)

ray.init(num_gpus=3)

hyperband=HyperBandScheduler(time_attr="training_iteration",reward_attr="mean_loss",max_t=100)

exp={
        'run':"my_class",
        'trial_resources':{
            "gpu":1
            },
        'repeat':3,
        'stop':{"training_iteration":100},
        'config':{
        "lr":lambda spec: 10**-(2*random.random()+2),
        "L2":lambda spec: 0.1*random.random()
    }
 }

tune.register_trainable("my_class", train_class)

tune.run_experiments({"my_Experiment":exp},scheduler=hyperband)



