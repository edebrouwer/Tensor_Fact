#plot time latents

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

import sys
sys.path.append("../")
from tensor_fact import tensor_fact

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import matplotlib
import matplotlib.pyplot as plt

def load_current_model(directory): #Function to load the saved model.
    loaded_mod=torch.load(directory+"best_model.pt")
    mod=tensor_fact(n_pat=loaded_mod["pat_lat.weight"].size(0),n_meas=loaded_mod["meas_lat.weight"].size(0),n_t=loaded_mod["time_lat.weight"].size(0),l_dim=loaded_mod["meas_lat.weight"].size(1),n_u=loaded_mod["beta_u"].size(0),n_w=loaded_mod["beta_w"].size(0))
    mod.double()
    mod.load_state_dict(loaded_mod)
    return(mod)

def plot_latent_times(directory):
    #mod=load_current_model(directory)
    mod=torch.load(directory+"best_model.pt")
    time_lats=mod["time_lat.weight"].detach().numpy()
    plt.plot(time_lats/np.std(time_lats,axis=0))
    plt.title("Latent time series")
    plt.savefig(directory+"latent_times.pdf")
    #print(mod.pat_lat.weight)

#One should give the path of the directory as argument.
if __name__=="__main__":
    plot_latent_times(directory=sys.argv[1:][0])
