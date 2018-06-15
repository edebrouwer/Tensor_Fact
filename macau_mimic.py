

import macau
import pandas as pd
import numpy as np

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

results=macau.macau(Y=df,Ytest=df_val,side=[cov_u,None,cov_t],num_latent=25,verbose=True,burnin=400,precision="adaptive",save_prefix="25_macau")



