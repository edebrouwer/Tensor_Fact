

import numpy as np
import sys
from sklearn.decomposition import PCA

dir_path=sys.argv[1:][0] #Should be like "./results_macau_70/"



def PCA_macau_samples(dir_path,idx_train=None,idx_val=None):
    sum_sim=np.load(dir_path+"sum_sim.npy").item()
    N_latents=sum_sim["N_latents"]
    N_samples=sum_sim["N_samples"]


    concat_lat=np.loadtxt(dir_path+str(N_latents)+"_macau-sample1-U1-latents.csv",delimiter=",")

    for n in np.linspace(10,N_samples,50,dtype='int'):
        concat_lat=np.concatenate((concat_lat,np.loadtxt(dir_path+str(N_latents)+"_macau-sample%d-U1-latents.csv"%n,delimiter=",")))

    if idx_train:
        concat_subset=concat_lat[idx_train,:]
    else:
        concat_subset=concat_lat

    pca=PCA()
    pca.fit(concat_subset.T)
    #print(np.cumsum(pca.explained_variance_ratio_))

    n_kept=np.min(np.where(np.cumsum(pca.explained_variance_ratio_)>0.9))
    pca=PCA(n_components=n_kept)
    pca.fit(concat_subset.T)

    pca_latents=pca.transform(concat_lat.T)

    np.save(dir_path+"pca_latents",pca_latents)
    print(n_kept)

    return(pca_latents[idx_train,:],pca_latents[idx_val,:])


if __name__=="__main__":
    PCA_macau_samples(dir_path)
