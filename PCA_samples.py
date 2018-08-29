

import numpy as np


dir_path=sys.argv[1:][0] #Should be like "./results_macau_70/"

sum_sim=np.load(dir_path+"sum_sim.npy")
N_latents=sum_sim{"N_latents"}
N_samples=sum_sim{"N_samples"}


concat_lat=np.loadtxt(dir_path+str(N_latents)+"_macau-sample1-U1-latents.csv",delimiter=",")

for n in np.linspace(10,N_samples,10,dtype='int')
    concat_lat=np.concatenate((concat_lat,np.loadtxt(dir_path+str(N_latents)+"_macau-sample%d-U1-latents.csv"%n,delimiter=",")))
