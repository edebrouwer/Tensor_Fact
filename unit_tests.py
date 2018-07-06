import numpy as np
import pandas as pd
import scipy.sparse
import macau
import itertools


num_latents=2

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

results=macau.macau(Y=df_train,Ytest=df_val,side=[None,None,None],num_latent=num_latents,verbose=True,burnin=400,nsamples=400,precision="adaptive",save_prefix=save_prefix)
