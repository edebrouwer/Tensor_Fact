import numpy as np

#loading_latent_matrices
dir_path="results_macau_16/"
file_path="16_macau_"
N=400

mean_lat_pat=0
mean_lat_meas=0
mean_lat_time=0
for n in range(N):
    mean_lat_pat+=np.loadtxt(dir_path+file_path+"sample%d-U1-latents.csv"%n,delimiter=",")
    mean_lat_meas+=np.loadtxt(dir_path+file_path+"sample%d-U2-latents.csv"%n,delimiter=",")
    mean_lat_time+=np.loadtxt(dir_path+file_path+"sample%d-U3-latents.csv"%n,delimiter=",")

mean_lat_pat/=N
mean_lat_meas/=N
mean_lat_time/=N

print(mean_lat_meas)
