import numpy as np
import pandas as pd
import scipy.sparse
import macau
import itertools

from tensor_utils import tensor_fact,TensorFactDataset, TensorFactDataset_ByPat, mod_select
from tensor_fact import train_model

import os
import shutil

parser=argparse.ArgumentParser(description="Longitudinal Tensor Factorization Unit Tests")

#Model parameters
parser.add_argument('--L2',default=0,type=float,help="L2 penalty (weight decay")
parser.add_argument('--lr',default=0.02,type=float,help="Learning rate of the optimizer")
parser.add_argument('--epochs',default=250,type=int,help="Number of epochs")
parser.add_argument('--latents',default=8,type=int,help="Number of latent dimensions")
parser.add_argument('--batch',default=65000,type=int,help="Number of samples per batch")

#Model selection
parser.add_argument('--DL',action='store_true',help="To switch to Deep Learning model")
parser.add_argument('--by_pat',action='store_true',help="To switch to by patient learning")
parser.add_argument('--death_label',action='store_true',help="Supervised training for the death labs")
parser.add_argument('--hard_split',action='store_true',help="To use the challenging validation splitting")
parser.add_argument('--XT',action='store_true',help="To use DL model on top of the product of latents")
#GPU args
parser.add_argument('--cuda',action='store_true')
parser.add_argument('--gpu_name',default='Titan',type=str,help="Name of the gpu to use for computation Titan or Tesla")
#Savings args
#parser.add_argument('--outfile',default="./",type=str,help="Path to save the models and outpus")

def macau_test:
    num_latents=2
    n_samples=800

    save_prefix="macau_unit_test"
    ## generating toy data
    A = np.random.randn(15, num_latents)
    B = np.random.randn(3, num_latents)
    C = np.random.randn(5, num_latents)

    idx = list( itertools.product(np.arange(A.shape[0]),
                                  np.arange(B.shape[0]),
                                  np.arange(C.shape[0])) )
    df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])


    df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])

    ## side information is again a sparse matrix

    df_train,df_val= macau.make_train_test_df(df,0.05)

    results=macau.macau(Y=df_train,Ytest=df_val,side=[None,None,None],num_latent=num_latents,verbose=True,burnin=400,nsamples=n_samples,precision="adaptive",save_prefix=save_prefix)


    str_dir="results_unittests/"
    if (not os.path.exists(str_dir)):
        os.makedirs(str_dir)
    else:
        #replace_prev=input("This configuration has already been run !Do you want to continue ? y/n")
        #if (replace_prev=="n"):
        #    raise ValueError("Aborted")
        shutil.rmtree(str_dir)
        os.makedirs(str_dir)

    files= os.listdir("./")
    for f in files:
        if (f.startswith(save_prefix)):
            shutil.move(f,str_dir)
    file_path=save_prefix
    N=n_samples

    mean_lat_pat=0
    mean_lat_meas=0
    mean_lat_time=0
    for n in range(1,N+1):
        mean_lat_pat+=np.loadtxt(str_dir+file_path+"-sample%d-U1-latents.csv"%n,delimiter=",")
        mean_lat_meas+=np.loadtxt(str_dir+file_path+"-sample%d-U2-latents.csv"%n,delimiter=",")
        mean_lat_time+=np.loadtxt(str_dir+file_path+"-sample%d-U3-latents.csv"%n,delimiter=",")

    mean_lat_pat/=N
    mean_lat_meas/=N
    mean_lat_time/=N

    np.save(str_dir+"mean_pat_latent.npy",mean_lat_pat)
    np.save(str_dir+"mean_pat_latent.npy",mean_lat_meas)
    np.save(str_dir+"mean_pat_latent.npy",mean_lat_time)


    print("Loaded")
    print("Patients Latents")
    print(mean_lat_pat.T)
    print("True Latents")
    print(A)
    print("Features Latents")
    print(mean_lat_meas.T)
    print("True Latents")
    print(B)
    print("Time Latents")
    print(mean_lat_time.T)
    print("True Latents")
    print(C)

def pytorch_test:
    num_latents=2

    save_prefix="macau_unit_test"
    ## generating toy data
    A = np.random.randn(15, num_latents)
    B = np.random.randn(3, num_latents)
    C = np.random.randn(5, num_latents)

    idx = list( itertools.product(np.arange(A.shape[0]),
                                  np.arange(B.shape[0]),
                                  np.arange(C.shape[0])) )
    df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])


    df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])

    cov=np.random.randn(15,4)
    cov_df=pd.DataFrame(cov)

    df.to_csv("Unit_tests_tensor.csv")
    cov_df.to_csv("Unit_test_cov.csv")

    dataloader, dataloader_val, mod, device,str_dir = mod_select(opt,tensor_path="Unit_tests_tensor",cov_path="Unit_test_cov")

    train_model(dataloader,dataloader_val,mod,device,str_dir,opt)


if __name__=="main":
    pytorch_test()
