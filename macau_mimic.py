import argparse
import macau
import pandas as pd
import numpy as np

import os

parser=argparse.ArgumentParser(description="Bayesian Tensor Factorization")


parser.add_argument('--latents',default=8,type=float,help="Number of latent dimensions")
parser.add_argument('--hard',action='store_true',help="Challenging validation split")

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

dir_path="~/Data/MIMIC/"
lab_short=pd.read_csv(dir_path+"lab_short_tensor_train.csv")
df=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

lab_short_val=pd.read_csv(dir_path+"lab_short_tensor_val.csv")
df_val=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
cov_u=lab_short.groupby("UNIQUE_ID").first()[cov_values].as_matrix()
cov_t=np.expand_dims(np.arange(101),axis=1)

print(cov_u.shape)
print(cov_t.shape)

results=macau.macau(Y=df,Ytest=df_val,side=[cov_u,None,cov_t],num_latent=num_latents,verbose=True,burnin=400,precision="adaptive",save_prefix=save_prefix)

files= os.listdir("./")
for f in files:
    if (f.startswith(save_prefix)):
        shutil.move(f,str_dir)
