
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

from tensor_fact import tensor_fact, TensorFactDataset

import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def load_current_model(): #Function to load the saved model.
        
        val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val.csv")
        mod=tensor_fact(n_pat=val_dataset.pat_num,n_meas=30,n_t=101,n_u=18,n_w=1)
        mod.double()
        mod.load_state_dict(torch.load("current_model.pt"))
        return(mod)
if __name__=="main":
    mod=load_current_model()
