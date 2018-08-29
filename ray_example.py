
import warnings
#warnings.filterwarnings("ignore", message="numpy.dtype size changed",RuntimeWarning)
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed",RuntimeWarning)

import random
import numpy as np
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

import os
import shutil

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

#@ray.remote(num_gpus=1)
class train_class(Trainable):
    def _setup(self):
        self.device=torch.device("cuda:0")
        mod_opt={'type':"plain_fact",'cov':False,'latents':20}
        
        #data_train=TensorFactDataset(csv_file_serie="complete_tensor_train1.csv",cov_path="complete_covariates")

        self.mod=model_selector(get_pinned_object(data_train),mod_opt,"cuda:0")
        #self.mod=model_selector(data_train,mod_opt,"cpu")

        self.dataloader = DataLoader(get_pinned_object(data_train),batch_size=65000,shuffle=True,num_workers=2)
        self.dataloader_val= DataLoader(get_pinned_object(data_val),batch_size=10000,shuffle=False)
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
        
        optimizer = torch.optim.Adam(self.mod.get_list_embeddings_params(), lr=l_r, weight_decay=self.config["L2"])  
        optimizer_w = torch.optim.Adam(self.mod.get_list_betas(), lr=l_r, weight_decay=self.config["L2_w"])
        
        criterion=nn.MSELoss()
        total_loss=0
        for idx, sampled_batch in enumerate(self.dataloader):    
            optimizer.zero_grad()
            optimizer_w.zero_grad()
            indexes=sampled_batch[:,1:4].to(torch.long).to(self.device)
            target=sampled_batch[:,-1].to(self.device)
            preds=self.mod.forward(indexes[:,0],indexes[:,1],indexes[:,2])
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            optimizer_w.step()
            total_loss+=loss
        mean_loss_computed=(total_loss.detach().cpu().numpy()/(idx+1))
        
        with torch.no_grad():
            loss_val=0
            for i_val,batch_val in enumerate(self.dataloader_val):
                indexes=batch_val[:,1:4].to(torch.long).to(self.device)
                target=batch_val[:,-1].to(self.device)
                pred_val=self.mod.forward(indexes[:,0],indexes[:,1],indexes[:,2])
                loss_val+=criterion(pred_val,target)
        rmse_val_loss_computed=(np.sqrt(loss_val.detach().cpu().numpy()/(i_val+1)))
        
        return TrainingResult(mean_loss=rmse_val_loss_computed,timesteps_this_iter=1)
    
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

#shutil.rmtree("/home/edward/ray_results/my_Experiment/")

ray.init(num_gpus=3,num_cpus=10)

data_train=pin_in_object_store(TensorFactDataset(csv_file_serie="complete_tensor_train1.csv",cov_path="complete_covariates"))
data_val=pin_in_object_store(TensorFactDataset(csv_file_serie="complete_tensor_val1.csv",cov_path="complete_covariates"))

tune.register_trainable("my_class", train_class)

hyperband=HyperBandScheduler(time_attr="timesteps_total",reward_attr="neg_mean_loss",max_t=100)

exp={
        'run':"my_class",
        'trial_resources':{
            "gpu":1
            },
        'repeat':50,
        'stop':{"training_iteration":80},
        'config':{
        "L2":lambda spec: 10**(-4*random.random()-6),
        "L2_w":lambda spec: 10**(-4*random.random()-6),
    }
 }

tune.run_experiments({"70_latents_2L2_nocov":exp},scheduler=hyperband)



