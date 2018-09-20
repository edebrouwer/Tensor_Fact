
import argparse
import macau
import pandas as pd
import numpy as np

import os
import shutil

from PCA_samples import Macau_PCA_samples

parser=argparse.ArgumentParser(description="Bayesian Tensor Factorization")


parser.add_argument('--latents',default=8,type=int,help="Number of latent dimensions")
parser.add_argument('--samples',default=400,type=int,help="Number of samples after burnin")
parser.add_argument('--burnin',default=400,type=int,help="Number of burnin samples")

opt=parser.parse_args()

files_path="~/Data/MIMIC/Clean_data/"
tensor_path=files_path+"complete_tensor.csv"
covariates_path=files_path+"complete_covariates.csv"


tensor=pd.read_csv(tensor_path)
covariates=pd.read_csv(covariates_path).sort_values(by="UNIQUE_ID").as_matrix()[:,1:]
tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")

num_patients=tensor["UNIQUE_ID"].nunique()

kf=KFold(n_splits=10)
kf_val=KFold(n_splits=5,random_state=42)
results_path=run_macau(num_latents=opt.latents,num_samples=opt.samples,num_burnin=opt.burnin,tensor=tensor,covariates=covariates)
#TEST LOOP
for train_val_index, test_index in kf.split(np.arange(num_patients)):

    #train classifier using ray.
    run_ray_logistic(results_path,tags,kf_val,train_val_index)

    #Get the best parameters
