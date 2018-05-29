#plot time latents

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from tensor_fact import tensor_fact, TensorFactDataset

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys

import matplotlib
import matplotlib.pyplot as plt

def load_current_model(directory="~/Projects/Tensor_Fact/trained_models/8_dim_500epochs_lr02/"): #Function to load the saved model.
    train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train.csv",file_path="~/Data/MIMIC/")
    mod=tensor_fact(n_pat=train_dataset.pat_num,n_meas=30,n_t=101,l_dim=8,n_u=18,n_w=1)
    mod.double()
    mod.load_state_dict(torch.load(directory+"current_model.pt"))
    return(mod)

def plot_latent_times(directory):
    mod=load_current_model(directory)
    plt.plot(mod.time_lat.weight.detach().numpy())
    plt.title("Latent time series")
    plt.savefig("latent_times.pdf")
    #print(mod.pat_lat.weight)

#One should give the path of the directory as argument.
if __name__=="__main__":
    plot_latent_times(directory=sys.argv[1:][0])
