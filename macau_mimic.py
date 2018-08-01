import argparse
import macau
import pandas as pd
import numpy as np

import os
import shutil

parser=argparse.ArgumentParser(description="Bayesian Tensor Factorization")


parser.add_argument('--latents',default=8,type=int,help="Number of latent dimensions")
parser.add_argument('--hard',action='store_true',help="Challenging validation split")
parser.add_argument('--samples',default=400,type=int,help="Number of samples after burnin")

opt=parser.parse_args()

num_latents=opt.latents
save_prefix=str(num_latents)+"_macau"
str_dir="results_macau_"+str(num_latents)+"/"
if (not os.path.exists(str_dir)):
    os.makedirs(str_dir)
else:
    replace_prev=input("This configuration has already been run !Do you want to continue ? y/n")
    if (replace_prev=="n"):
        raise ValueError("Aborted")
    shutil.rmtree(str_dir)
    os.makedirs(str_dir)

dir_path="~/Data/MIMIC/"
lab_short=pd.read_csv(dir_path+"complete_tensor.csv")
df=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

lab_short_val=pd.read_csv(dir_path+"complete_tensor_val.csv")
df_val=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

#cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
#cov_u=lab_short.groupby("UNIQUE_ID").first()[cov_values].as_matrix()
cov_t=np.expand_dims(np.arange(97),axis=1)

cov_u=pd.read_csv(dir_path+"complete_covariates.csv").as_matrix()[:,1:]


print(cov_u.shape)
print(cov_t.shape)

results=macau.macau(Y=df,Ytest=df_val,side=[cov_u,None,cov_t],num_latent=num_latents,verbose=True,burnin=400,nsamples=opt.samples,precision="adaptive",save_prefix=save_prefix)

print("TEST RMSE : "+str(results.rmse_test))

files= os.listdir("./")
for f in files:
    if (f.startswith(save_prefix)):
        shutil.move(f,str_dir)

import progressbar

#EXTRACT THE LATENTS OF MACAU AND COMPUTES THE MEAN.

#loading_latent_matrices
file_path=save_prefix
N=opt.samples

mean_lat_pat=0
mean_lat_meas=0
mean_lat_time=0
for n in progressbar.progressbar(range(1,N+1)):
    mean_lat_pat+=np.loadtxt(str_dir+file_path+"-sample%d-U1-latents.csv"%n,delimiter=",")
    #mean_lat_meas+=np.loadtxt(dir_path+file_path+"sample%d-U2-latents.csv"%n,delimiter=",")
    mean_lat_time+=np.loadtxt(str_dir+file_path+"-sample%d-U3-latents.csv"%n,delimiter=",")

mean_lat_pat/=N
np.save(str_dir+"mean_pat_latent.npy",mean_lat_pat)
np.save(str_dir+"mean_time_latent.npy",mean_lat_time)

print("Loaded")
