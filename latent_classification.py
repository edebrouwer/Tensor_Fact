#Death classification
import torch
import pandas as pd
import numpy as np
from sklearn import svm
from tensor_fact_test import load_current_model
from sklearn.model_selection import cross_val_score


file_path="./trained_models/8_dim_250_lr02/"
latent_pat=torch.load(file_path+"current_model.pt")["pat_lat.weight"].numpy()
tags=pd.read_csv("~/Data/MIMIC/death_tag_tensor.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

mean_res=[]
std_res=[]
for c in [0.0001,0.001,0.01]:

    print("Baseline : "+str(1-np.sum(tag_mat)/tag_mat.shape[0]))

    clf=svm.SVC(C=c,class_weight="balanced")
    scores=cross_val_score(clf,latent_pat,tag_mat,cv=10)
    print("C values : "+str(c))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    mean_res=mean_res+scores.mean()
    std_res=std_res+scores.std()
