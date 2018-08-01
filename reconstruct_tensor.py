

import numpy as np

if "macau" in file_path:
    latent_pat=np.load(file_path+"mean_pat_latent.npy").T
    latent_times=np.load(file_path+"mean_time_latent.npy").T
    latent_feat=np.load(file_path+"mean_time_latent.npy").T

    print(latent_pat.shape())
