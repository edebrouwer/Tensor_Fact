

import macau
import pandas as import pd

dir_path="~/Data/MIMIC/"
lab_short=pd.read_csv(dir_path+"lab_short_tensor_train.csv")
df=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

lab_short_val=pd.read_csv(dir_path+"lab_short_tensor_val.csv")
df_val=lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]]

cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
covariates=lab_short.groupby("UNIQUE_ID").first()[cov_values].as_matrix()

results=macau.macau(Y=df,Ytest=df_train,side=[covariates,None,None],num_latents=8,verbose=True,burnin=800)
