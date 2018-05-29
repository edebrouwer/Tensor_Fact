#Death classification
import torch
import pandas as pd
from sklearn import svm
from tensor_fact_test import load_current_model

file_path="./trained_models/8_dim_500epochs_lr02"
latent_pat=torch.load(file_path+"current_model.pt")["pat_lat.weight"]
tags=pd.read_csv("~/Data/MIMIC/death_tag_tensor.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()
