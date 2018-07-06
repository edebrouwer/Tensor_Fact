import numpy as np
import pandas as pd
import scipy.sparse
import macau
import itertools


num_latents=2
n_samples=400

save_prefix="unit_test"
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

df_train,df_val= macau.make_train_test_df(df,0.2)

results=macau.macau(Y=df_train,Ytest=df_val,side=[None,None,None],num_latent=num_latents,verbose=True,burnin=400,nsamples=n_samples,precision="adaptive",save_prefix=save_prefix)

str_dir="results_unittests/"
if (not os.path.exists(str_dir)):
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
for n in progressbar.progressbar(range(1,N+1)):
    mean_lat_pat+=np.loadtxt(str_dir+file_path+"-sample%d-U1-latents.csv"%n,delimiter=",")
    mean_lat_meas+=np.loadtxt(dir_path+file_path+"sample%d-U2-latents.csv"%n,delimiter=",")
    mean_lat_time+=np.loadtxt(dir_path+file_path+"sample%d-U3-latents.csv"%n,delimiter=",")

mean_lat_pat/=N
mean_lat_meas/=N
mean_lat_time/=N

np.save(str_dir+"mean_pat_latent.npy",mean_lat_pat)
np.save(str_dir+"mean_pat_latent.npy",mean_lat_meas)
np.save(str_dir+"mean_pat_latent.npy",mean_lat_time)


print("Loaded")
print(mean_lat_pat)
