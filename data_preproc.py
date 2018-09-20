# Merging of the different MIMIC data sources

##### This file takes as inputs :

# -LAB_processed (from notebook LabEvents) with the pre-selected and cleaned lab measurements of the patients
#
# -INPUTS_processed (from notebook Admissions) with the pre-selected and cleaned inputs to the patients
#
# -Admissions_processed (from the notebook Admissions) with the death label of the patients
#
# -Diagnoses_ICD with the ICD9 codes of each patient.
#
# ##### This notebook outputs :
#
# -death_tags.csv. A dataframe with the patient id and the corresponding death label
#
# -complete_tensor_csv. A dataframe containing all the measurments in tensor version.
#
# -complete_tensor_train.csv. A dataframe containing all the training measurments in tensor version.
#
# -complete_tensor_val.csv. A dataframe containing all the validation measurments in tensor version.
#
# -complete_covariates.csv. A dataframe with the ICD9 covariates codes (binary) of each patient index.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import datetime
from datetime import timedelta
import numpy as np

file_path="~/Data/MIMIC/"
outfile_path="~/Data/MIMIC/Clean_data/"

lab_df=pd.read_csv(file_path+"LAB_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","VALUENUM","LABEL"]]
inputs_df=pd.read_csv(file_path+"INPUTS_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","AMOUNT","LABEL"]]
outputs_df=pd.read_csv(file_path+"OUTPUTS_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","VALUE","LABEL"]]
presc_df=pd.read_csv(file_path+"PRESCRIPTIONS_processed.csv")[["SUBJECT_ID","HADM_ID","CHARTTIME","DOSE_VAL_RX","DRUG"]]



#Process names of columns to have the same everywhere.

#Change the name of amount. Valuenum for every table
inputs_df["VALUENUM"]=inputs_df["AMOUNT"]
inputs_df.head()
inputs_df=inputs_df.drop(columns=["AMOUNT"]).copy()

#Change the name of amount. Valuenum for every table
outputs_df["VALUENUM"]=outputs_df["VALUE"]
outputs_df=outputs_df.drop(columns=["VALUE"]).copy()

#Change the name of amount. Valuenum for every table
presc_df["VALUENUM"]=presc_df["DOSE_VAL_RX"]
presc_df=presc_df.drop(columns=["DOSE_VAL_RX"]).copy()
presc_df["LABEL"]=presc_df["DRUG"]
presc_df=presc_df.drop(columns=["DRUG"]).copy()


#Tag to distinguish between lab and inputs events
inputs_df["Origin"]="Inputs"
lab_df["Origin"]="Lab"
outputs_df["Origin"]="Outputs"
presc_df["Origin"]="Prescriptions"


#merge both dfs.
merged_df1=(inputs_df.append(lab_df)).reset_index()
merged_df2=(merged_df1.append(outputs_df)).reset_index()
merged_df2.drop(columns="level_0",inplace=True)
merged_df=(merged_df2.append(presc_df)).reset_index()

#merged_df=lab_df.reset_index()

#Check that all labels have different names.
assert(merged_df["LABEL"].nunique()==(inputs_df["LABEL"].nunique()+lab_df["LABEL"].nunique()+outputs_df["LABEL"].nunique()+presc_df["LABEL"].nunique()))

#Set the reference time as the lowest chart time for each admission.
merged_df['CHARTTIME']=pd.to_datetime(merged_df["CHARTTIME"], format='%Y-%m-%d %H:%M:%S')
ref_time=merged_df.groupby("HADM_ID")["CHARTTIME"].min()

merged_df_1=pd.merge(ref_time.to_frame(name="REF_TIME"),merged_df,left_index=True,right_on="HADM_ID")
merged_df_1["TIME_STAMP"]=merged_df_1["CHARTTIME"]-merged_df_1["REF_TIME"]
assert(len(merged_df_1.loc[merged_df_1["TIME_STAMP"]<timedelta(hours=0)].index)==0)

#Create a label code (int) for the labels.
label_dict=dict(zip(list(merged_df_1["LABEL"].unique()),range(len(list(merged_df_1["LABEL"].unique())))))
merged_df_1["LABEL_CODE"]=merged_df_1["LABEL"].map(label_dict)

merged_df_short=merged_df_1[["HADM_ID","VALUENUM","TIME_STAMP","LABEL_CODE","Origin"]]

#To do : store the label dictionnary in a csv file.

label_dict_df=pd.Series(merged_df_1["LABEL"].unique()).reset_index()
label_dict_df.columns=["index","LABEL"]
label_dict_df["LABEL_CODE"]=label_dict_df["LABEL"].map(label_dict)
label_dict_df.drop(columns=["index"],inplace=True)
label_dict_df.to_csv(outfile_path+"label_dict.csv")

#### Time binning of the data
#First we select the data up to a certain time limit (48 hours)

#Now only select values within 48 hours.
merged_df_short=merged_df_short.loc[(merged_df_short["TIME_STAMP"]<timedelta(hours=48))]
print("Number of patients considered :"+str(merged_df_short["HADM_ID"].nunique()))

#Plot the number of "hits" based on the binning. That is, the number of measurements falling into the same bin in function of the number of bins
bins_num=range(1,10)
merged_df_short_binned=merged_df_short.copy()
hits_vec=[]
for bin_k in bins_num:
    time_stamp_str="TIME_STAMP_Bin_"+str(bin_k)
    merged_df_short_binned[time_stamp_str]=round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36)).astype(int)
    hits_prop=merged_df_short_binned.duplicated(subset=["HADM_ID","LABEL_CODE",time_stamp_str]).sum()/len(merged_df_short_binned.index)
    hits_vec+=[hits_prop]
# plt.plot(bins_num,hits_vec)
# plt.title("Percentage of hits in function of the binning factor")
# plt.xlabel("Number of bins/hour")
# plt.ylabel("% of hits")
# plt.show()

#We choose 2 bins per hour. We now need to aggregate the data in different ways.
bin_k=2
merged_df_short["TIME"]=round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36)).astype(int)

#For lab, we have to average the duplicates.
lab_subset=merged_df_short.loc[merged_df_short["Origin"]=="Lab",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
lab_subset["KEY_ID"]=lab_subset["HADM_ID"].astype(str)+"/"+lab_subset["TIME"].astype(str)+"/"+lab_subset["LABEL_CODE"].astype(str)
lab_subset["VALUENUM"]=lab_subset["VALUENUM"].astype(float)

lab_subset_s=lab_subset.groupby("KEY_ID")["VALUENUM"].mean().to_frame().reset_index()

lab_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
lab_s=pd.merge(lab_subset,lab_subset_s,on="KEY_ID")
assert(not lab_s.isnull().values.any())

#For inputs, we have to sum the duplicates.
input_subset=merged_df_short.loc[merged_df_short["Origin"]=="Inputs",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
input_subset["KEY_ID"]=input_subset["HADM_ID"].astype(str)+"/"+input_subset["TIME"].astype(str)+"/"+input_subset["LABEL_CODE"].astype(str)
input_subset["VALUENUM"]=input_subset["VALUENUM"].astype(float)

input_subset_s=input_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

input_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
input_s=pd.merge(input_subset,input_subset_s,on="KEY_ID")
assert(not input_s.isnull().values.any())

#For outpus, we have to sum the duplicates as well.
output_subset=merged_df_short.loc[merged_df_short["Origin"]=="Outputs",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
output_subset["KEY_ID"]=output_subset["HADM_ID"].astype(str)+"/"+output_subset["TIME"].astype(str)+"/"+output_subset["LABEL_CODE"].astype(str)
output_subset["VALUENUM"]=output_subset["VALUENUM"].astype(float)

output_subset_s=output_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

output_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
output_s=pd.merge(output_subset,output_subset_s,on="KEY_ID")
assert(not output_s.isnull().values.any())

#For prescriptions, we have to sum the duplicates as well.
presc_subset=merged_df_short.loc[merged_df_short["Origin"]=="Prescriptions",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
presc_subset["KEY_ID"]=presc_subset["HADM_ID"].astype(str)+"/"+presc_subset["TIME"].astype(str)+"/"+presc_subset["LABEL_CODE"].astype(str)
presc_subset["VALUENUM"]=presc_subset["VALUENUM"].astype(float)

presc_subset_s=presc_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

presc_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
presc_s=pd.merge(presc_subset,presc_subset_s,on="KEY_ID")
assert(not presc_s.isnull().values.any())

#Now remove the duplicates/
lab_s=(lab_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
input_s=(input_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
output_s=(output_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
presc_s=(presc_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()

#We append both subsets together to form the complete dataframe
complete_df1=lab_s.append(input_s)
complete_df2=complete_df1.append(output_s)
complete_df=complete_df2.append(presc_s)


assert(sum(complete_df.duplicated(subset=["HADM_ID","LABEL_CODE","TIME"])==True)==0) #Check if no duplicates anymore.

# We remove patients with less than 50 observations.
id_counts=complete_df.groupby("HADM_ID").count()
id_list=list(id_counts.loc[id_counts["TIME"]<50].index)
complete_df=complete_df.drop(complete_df.loc[complete_df["HADM_ID"].isin(id_list)].index).copy()

#We also choose 10 bins per hour. We now need to aggregate the data in different ways.
bin_k=10
merged_df_short["TIME"]=round(merged_df_short_binned["TIME_STAMP"].dt.total_seconds()*bin_k/(100*36))

#For lab, we have to average the duplicates.
lab_subset=merged_df_short.loc[merged_df_short["Origin"]=="Lab",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
lab_subset["KEY_ID"]=lab_subset["HADM_ID"].astype(str)+"/"+lab_subset["TIME"].astype(str)+"/"+lab_subset["LABEL_CODE"].astype(str)
lab_subset["VALUENUM"]=lab_subset["VALUENUM"].astype(float)

lab_subset_s=lab_subset.groupby("KEY_ID")["VALUENUM"].mean().to_frame().reset_index()

lab_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
lab_s=pd.merge(lab_subset,lab_subset_s,on="KEY_ID")
assert(not lab_s.isnull().values.any())

#For inputs, we have to sum the duplicates.
input_subset=merged_df_short.loc[merged_df_short["Origin"]=="Inputs",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
input_subset["KEY_ID"]=input_subset["HADM_ID"].astype(str)+"/"+input_subset["TIME"].astype(str)+"/"+input_subset["LABEL_CODE"].astype(str)
input_subset["VALUENUM"]=input_subset["VALUENUM"].astype(float)

input_subset_s=input_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

input_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
input_s=pd.merge(input_subset,input_subset_s,on="KEY_ID")
assert(not input_s.isnull().values.any())

#For outpus, we have to sum the duplicates as well.
output_subset=merged_df_short.loc[merged_df_short["Origin"]=="Outputs",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
output_subset["KEY_ID"]=output_subset["HADM_ID"].astype(str)+"/"+output_subset["TIME"].astype(str)+"/"+output_subset["LABEL_CODE"].astype(str)
output_subset["VALUENUM"]=output_subset["VALUENUM"].astype(float)

output_subset_s=output_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

output_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
output_s=pd.merge(output_subset,output_subset_s,on="KEY_ID")
assert(not output_s.isnull().values.any())

#For prescriptions, we have to sum the duplicates as well.
presc_subset=merged_df_short.loc[merged_df_short["Origin"]=="Prescriptions",["HADM_ID","TIME","LABEL_CODE","VALUENUM"]]
presc_subset["KEY_ID"]=presc_subset["HADM_ID"].astype(str)+"/"+presc_subset["TIME"].astype(str)+"/"+presc_subset["LABEL_CODE"].astype(str)
presc_subset["VALUENUM"]=presc_subset["VALUENUM"].astype(float)

presc_subset_s=presc_subset.groupby("KEY_ID")["VALUENUM"].sum().to_frame().reset_index()

presc_subset.rename(inplace=True,columns={"VALUENUM":"ExVALUENUM"})
presc_s=pd.merge(presc_subset,presc_subset_s,on="KEY_ID")
assert(not presc_s.isnull().values.any())

#Now remove the duplicates/
lab_s=(lab_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
input_s=(input_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
output_s=(output_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()
presc_s=(presc_s.drop_duplicates(subset=["HADM_ID","LABEL_CODE","TIME"]))[["HADM_ID","TIME","LABEL_CODE","VALUENUM"]].copy()

#We append both subsets together to form the complete dataframe
complete_df1=lab_s.append(input_s)
complete_df2=complete_df1.append(output_s)
complete_df10=complete_df2.append(presc_s)


assert(sum(complete_df10.duplicated(subset=["HADM_ID","LABEL_CODE","TIME"])==True)==0) #Check if no duplicates anymore.

# We remove patients with less than 50 observations.
id_counts=complete_df10.groupby("HADM_ID").count()
id_list=list(id_counts.loc[id_counts["TIME"]<50].index)
complete_df10=complete_df10.drop(complete_df10.loc[complete_df10["HADM_ID"].isin(id_list)].index).copy()


#SAPSII data
#saps=pd.read_csv(file_path+'saps2.csv')
#valid_hadm_id=complete_df["HADM_ID"].unique()
#saps=saps.loc[saps["hadm_id"].isin(list(valid_hadm_id))].copy()
#saps["HADM_ID"]=saps["hadm_id"]
#saps.head()

#saps["SUM_score"]=saps[[ 'hr_score', 'sysbp_score', 'temp_score', 'pao2fio2_score','uo_score', 'bun_score', 'wbc_score', 'potassium_score', 'sodium_score','bicarbonate_score', 'bilirubin_score', 'gcs_score']].sum(axis=1)
#saps["X"]=-7.7631 + 0.0737 * saps["SUM_score"] + 0.9971 * (np.log(saps["SUM_score"] + 1))
#saps["PROB"]=np.exp(saps["X"])/(1+np.exp(saps["X"]))

#saps_death=pd.merge(death_tags_df,saps,on="HADM_ID")
#y_pred=np.array(saps_death["PROB"])
#y_pred_full=np.array(saps_death["sapsii_prob"])
#y=np.array(saps_death["DEATHTAG"])

#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(y,y_pred))
#print(roc_auc_score(y,y_pred_full))


# Dataframe creation for Tensor Decomposition

#Creation of a unique index for the admissions id.

#Creation of a unique index
unique_ids=np.arange(complete_df["HADM_ID"].nunique())
np.random.shuffle(unique_ids)
d=dict(zip(complete_df["HADM_ID"].unique(),unique_ids))

Unique_id_dict=pd.Series(complete_df["HADM_ID"].unique()).reset_index().copy()
Unique_id_dict.columns=["index","HADM_ID"]
Unique_id_dict["UNIQUE_ID"]=Unique_id_dict["HADM_ID"].map(d)
Unique_id_dict.to_csv(outfile_path+"UNIQUE_ID_dict.csv")

### Death tags data set

admissions=pd.read_csv(file_path+"Admissions_processed.csv")
death_tags_s=admissions.groupby("HADM_ID")["DEATHTAG"].unique().astype(int).to_frame().reset_index()
death_tags_df=death_tags_s.loc[death_tags_s["HADM_ID"].isin(complete_df["HADM_ID"])].copy()
death_tags_df["UNIQUE_ID"]=death_tags_df["HADM_ID"].map(d)
death_tags_df.sort_values(by="UNIQUE_ID",inplace=True)
death_tags_df.to_csv(outfile_path+"complete_death_tags.csv")

### Tensor Dataset
complete_df["UNIQUE_ID"]=complete_df["HADM_ID"].map(d)

#ICD9 codes
ICD_diag=pd.read_csv(file_path+"DIAGNOSES_ICD.csv")

main_diag=ICD_diag.loc[(ICD_diag["SEQ_NUM"]==1)]
complete_tensor=pd.merge(complete_df,main_diag[["HADM_ID","ICD9_CODE"]],on="HADM_ID")

#Only select the first 3 digits of each ICD9 code.
complete_tensor["ICD9_short"]=complete_tensor["ICD9_CODE"].astype(str).str[:3]
#Check that all codes are 3 digits long.
str_len=complete_tensor["ICD9_short"].str.len()
assert(str_len.loc[str_len!=3].count()==0)

#Finer encoding (3 digits)
hot_encodings=pd.get_dummies(complete_tensor["ICD9_short"])
complete_tensor[hot_encodings.columns]=hot_encodings
complete_tensor_nocov=complete_tensor[["UNIQUE_ID","LABEL_CODE","TIME"]+["VALUENUM"]].copy()

complete_tensor_nocov.rename(columns={"TIME":"TIME_STAMP"},inplace=True)

### Normalization of the data (N(0,1))

#Add a column with the mean and std of each different measurement type and then normalize them.
d_mean=dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].mean())
complete_tensor_nocov["MEAN"]=complete_tensor_nocov["LABEL_CODE"].map(d_mean)
d_std=dict(complete_tensor_nocov.groupby("LABEL_CODE")["VALUENUM"].std())
complete_tensor_nocov["STD"]=complete_tensor_nocov["LABEL_CODE"].map(d_std)
complete_tensor_nocov["VALUENORM"]=(complete_tensor_nocov["VALUENUM"]-complete_tensor_nocov["MEAN"])/complete_tensor_nocov["STD"]

### Train-Validation-Test split
#Random sampling

#Split training_validation_test sets RANDOM DIVISION.

df_train,df_test =train_test_split(complete_tensor_nocov,test_size=0.1)

#Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert(len(df_test.loc[~df_test["UNIQUE_ID"].isin(df_train["UNIQUE_ID"])].index)==0)
assert(len(df_test.loc[~df_test["LABEL_CODE"].isin(df_train["LABEL_CODE"])].index)==0)

#First train_val fold
df_train1,df_val1 =train_test_split(df_train,test_size=0.2)

#Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert(len(df_val1.loc[~df_val1["UNIQUE_ID"].isin(df_train1["UNIQUE_ID"])].index)==0)
assert(len(df_val1.loc[~df_val1["LABEL_CODE"].isin(df_train1["LABEL_CODE"])].index)==0)

#Second train_val fold
df_train2,df_val2 =train_test_split(df_train,test_size=0.2)

#Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert(len(df_val2.loc[~df_val2["UNIQUE_ID"].isin(df_train2["UNIQUE_ID"])].index)==0)
assert(len(df_val2.loc[~df_val2["LABEL_CODE"].isin(df_train2["LABEL_CODE"])].index)==0)

#Third train_val fold
df_train3,df_val3 =train_test_split(df_train,test_size=0.2)

#Make sure that patients of the test set have instances in the training set. (same with labels but this should be nearly certain)
assert(len(df_val3.loc[~df_val3["UNIQUE_ID"].isin(df_train3["UNIQUE_ID"])].index)==0)
assert(len(df_val3.loc[~df_val3["LABEL_CODE"].isin(df_train3["LABEL_CODE"])].index)==0)

#Save locally.
complete_tensor_nocov.to_csv(outfile_path+"complete_tensor.csv") #Full data
df_train1.to_csv(outfile_path+"complete_tensor_train1.csv") #Train data
df_val1.to_csv(outfile_path+"complete_tensor_val1.csv") #Validation data
df_train2.to_csv(outfile_path+"complete_tensor_train2.csv") #Train data
df_val2.to_csv(outfile_path+"complete_tensor_val2.csv") #Validation data
df_train3.to_csv(outfile_path+"complete_tensor_train3.csv") #Train data
df_val3.to_csv(outfile_path+"complete_tensor_val3.csv") #Validation data
df_test.to_csv(outfile_path+"complete_tensor_test.csv") #Test data

#We create a data set with the covariates
covariates=complete_tensor.groupby("UNIQUE_ID").nth(0)[list(hot_encodings.columns)]
covariates.to_csv(outfile_path+"complete_covariates.csv") #save locally

## Creation of the dataset for LSTM operation

#We split the data patient-wise and provide imputation methods.

#Unique_ids of train and test
test_prop=0.1
val_prop=0.2
sorted_unique_ids=np.sort(unique_ids)
train_unique_ids=sorted_unique_ids[:int((1-test_prop)*(1-val_prop)*len(unique_ids))]
val_unique_ids=sorted_unique_ids[int((1-test_prop)*(1-val_prop)*len(unique_ids)):int((1-test_prop)*len(unique_ids))]
test_unique_ids=sorted_unique_ids[int((1-test_prop)*len(unique_ids)):]

death_tags_train_df=death_tags_df.loc[death_tags_df["UNIQUE_ID"].isin(list(train_unique_ids))].sort_values(by="UNIQUE_ID")
death_tags_val_df=death_tags_df.loc[death_tags_df["UNIQUE_ID"].isin(list(val_unique_ids))].sort_values(by="UNIQUE_ID")
death_tags_test_df=death_tags_df.loc[death_tags_df["UNIQUE_ID"].isin(list(test_unique_ids))].sort_values(by="UNIQUE_ID")

death_tags_train_df.to_csv(outfile_path+"LSTM_death_tags_train.csv")
death_tags_val_df.to_csv(outfile_path+"LSTM_death_tags_val.csv")
death_tags_test_df.to_csv(outfile_path+"LSTM_death_tags_test.csv")

#Create a segmented tensor (by patients)
complete_tensor_train=complete_tensor_nocov.loc[complete_tensor_nocov["UNIQUE_ID"].isin(list(train_unique_ids))].sort_values(by="UNIQUE_ID")
complete_tensor_val=complete_tensor_nocov.loc[complete_tensor_nocov["UNIQUE_ID"].isin(list(val_unique_ids))].sort_values(by="UNIQUE_ID")
complete_tensor_test=complete_tensor_nocov.loc[complete_tensor_nocov["UNIQUE_ID"].isin(list(test_unique_ids))].sort_values(by="UNIQUE_ID")

complete_tensor_train.to_csv(outfile_path+"LSTM_tensor_train.csv")
complete_tensor_val.to_csv(outfile_path+"LSTM_tensor_val.csv")
complete_tensor_test.to_csv(outfile_path+"LSTM_tensor_test.csv")

covariates_train=covariates.loc[covariates.index.isin(train_unique_ids)].sort_index()
covariates_val=covariates.loc[covariates.index.isin(val_unique_ids)].sort_index()
covariates_test=covariates.loc[covariates.index.isin(test_unique_ids)].sort_index()

covariates_train.to_csv(outfile_path+"LSTM_covariates_train.csv") #save locally
covariates_val.to_csv(outfile_path+"LSTM_covariates_val.csv") #save locally
covariates_test.to_csv(outfile_path+"LSTM_covariates_test.csv") #save locally

#Vector containing the mean_values of each dimension.
mean_dims=complete_tensor_train.groupby("LABEL_CODE")["MEAN"].mean()
mean_dims.to_csv(outfile_path+"mean_features.csv")
