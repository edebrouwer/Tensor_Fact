import numpy as np
import progressbar

#loading_latent_matrices
dir_path="results_macau_25/"
file_path="25_macau-"
N=400

mean_lat_pat=0
mean_lat_meas=0
mean_lat_time=0
for n in progressbar.progressbar(range(1,N+1)):
    mean_lat_pat+=np.loadtxt(dir_path+file_path+"sample%d-U1-latents.csv"%n,delimiter=",")
    #mean_lat_meas+=np.loadtxt(dir_path+file_path+"sample%d-U2-latents.csv"%n,delimiter=",")
    #mean_lat_time+=np.loadtxt(dir_path+file_path+"sample%d-U3-latents.csv"%n,delimiter=",")

mean_lat_pat/=N
np.save(dir_path+"mean_pat_latent.npy",mean_lat_pat)

print("Loaded")



