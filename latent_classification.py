#Death classification
import torch
import pandas as pd
from sklearn import svm
from tensor_fact_test import load_current_model

mod=load_current_model()
latent_pat=mod.pat_lat.weight
tags=pd.read_csv("~/Data/MIMIC/death_tag_tensor.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()
