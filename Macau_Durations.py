
import argparse
import macau
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
import os
import shutil
from macau_mimic import run_macau
from PCA_samples import PCA_macau_samples
from ray_logistic import run_ray_logistic
parser=argparse.ArgumentParser(description="Bayesian Tensor Factorization")

parser.add_argument('--latents',default=8,type=int,help="Number of latent dimensions")
parser.add_argument('--samples',default=400,type=int,help="Number of samples after burnin")
parser.add_argument('--burnin',default=400,type=int,help="Number of burnin samples")

opt=parser.parse_args()

log_name="Nested_ICU"
files_path="~/Data/MIMIC/Clean_data/"
tensor_path=files_path+"complete_tensor.csv"
covariates_path=files_path+"complete_covariates.csv"


tensor=pd.read_csv(tensor_path)
covariates=pd.read_csv(covariates_path).sort_values(by="UNIQUE_ID").values[:,1:]
tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")[["DEATHTAG","UNIQUE_ID"]].values[:,0]

num_patients=tensor["UNIQUE_ID"].nunique()

results_path=run_macau(num_latents=opt.latents,num_samples=opt.samples,num_burnin=opt.burnin,tensor=tensor,covariates=covariates)

kf=KFold(n_splits=5)
train_idx, test_idx = train_test_split(np.arange(tags.shape[0]))
run_ray_logistic(results_path,tags,kf,idx=train_idx,log_name=log_name)


    #Should now retrieve the best parameters for each fold and retrain fully on the train+validation set.
