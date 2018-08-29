

import numpy as np
import sys
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

file_path=sys.argv[1:][0]

if "macau" in file_path:

    true_tensor=pd.read_csv("~/Data/MIMIC/complete_tensor_train1.csv")[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]
    true_tensor_val=pd.read_csv("~/Data/MIMIC/complete_tensor_val1.csv")[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

    dict_sum=np.load(file_path+"sum_sim.npy").item()
    N_samples=dict_sum["N_samples"]
    N_latents=dict_sum["N_latents"]


    print("Number of samples in this simulation is {}".format(N_samples))

    val_idx=true_tensor_val.values[:,:3]
 
    mse=0
    num_lat=true_tensor["UNIQUE_ID"].nunique()
    num_feat=true_tensor["LABEL_CODE"].nunique()
    num_times=true_tensor["TIME_STAMP"].nunique()
    reconstruct_sum=np.zeros((num_lat,num_feat,num_times))
    for i in range(N_samples):
        print(i)
        latent_pat=np.loadtxt(file_path+str(N_latents)+"_macau-sample"+str(i+1)+"-U1-latents.csv",delimiter=",").T
        latent_times=np.loadtxt(file_path+str(N_latents)+"_macau-sample"+str(i+1)+"-U3-latents.csv",delimiter=",").T
        latent_feat=np.loadtxt(file_path+str(N_latents)+"_macau-sample"+str(i+1)+"-U2-latents.csv",delimiter=",").T

        print("Einsum")
        reconstructed_tensor=np.einsum('il,jl,kl->ijk',latent_pat,latent_feat,latent_times)
        print("Built learnt tensor with shape{}".format(reconstructed_tensor.shape))

        reconstruct_sum+=reconstructed_tensor
    
    reconstruct_mean=reconstruct_sum/N_samples
    reconstructed_val=reconstruct_mean[val_idx[:,0].astype(int),val_idx[:,1].astype(int),val_idx[:,2].astype(int)]
    mse=np.mean((reconstructed_val-true_tensor_val.values[:,3])**2)
    print("MSE of the validation set is {}".format(mse))
    print("RMSE of the validation set is {}".format(np.sqrt(mse)))

    i_pat=7873
    i_feat=19
    
    latent_pat=np.load(file_path+"mean_pat_latent.npy").T
    latent_times=np.load(file_path+"mean_time_latent.npy").T
    latent_feat=np.load(file_path+"mean_feat_latent.npy").T

    reconstructed_tensor=np.einsum('il,jl,kl->ijk',latent_pat,latent_feat,latent_times)

    reconstructed_series=reconstructed_tensor[i_pat,i_feat,:]
    true_samples=np.zeros(reconstructed_tensor.shape[2])

    true_samples=true_tensor.loc[(true_tensor["LABEL_CODE"]==i_feat)&(true_tensor["UNIQUE_ID"]==i_pat)].values
    true_series=true_samples[:,[2,3]]

    true_samples_val=true_tensor_val.loc[(true_tensor_val["LABEL_CODE"]==i_feat)&(true_tensor_val["UNIQUE_ID"]==i_pat)].values
    true_series_val=true_samples_val[:,[2,3]]

    print(true_series)
    plt.plot(np.arange(reconstructed_tensor.shape[2]),reconstructed_series)
    plt.scatter(true_series[:,0],true_series[:,1],label="train samples")
    plt.scatter(true_series_val[:,0],true_series_val[:,1],label="validation samples")
    plt.legend()
    plt.savefig(file_path+"reconstruction_curves.pdf")
