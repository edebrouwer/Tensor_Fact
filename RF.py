#Death classification
import torch
import pandas as pd
import numpy as np
from tensor_fact_test import load_current_model
from sklearn.model_selection import cross_val_score
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
from multiprocessing.pool import Pool as PoolParent
from multiprocessing import Process, Pool

from sklearn.ensemble import RandomForestClassifier
import sys

from functools import partial

file_path=sys.argv[1:][0]

if "macau" in file_path:
    latent_pat=np.load(file_path+"mean_pat_latent.npy").T
else:
    latent_pat=torch.load(file_path+"best_model.pt")["pat_lat.weight"].cpu().numpy() #latents without covariates
print(latent_pat.shape)

tags=pd.read_csv("~/Data/MIMIC/death_tag_tensor.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

print("Data is Loaded")


def roc_comp(train_test):
    #print("roc_comp with C parameter ="+str(clf.get_params()["C"]))
    clf=RandomForestClassifier(max_depth=2,random_state=0)
    probas_=clf.fit(latent_pat[train_test[0]],tag_mat[train_test[0]]).predict_proba(latent_pat[train_test[1]])
    fpr, tpr, thresholds = roc_curve(tag_mat[train_test[1]], probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return([fpr,tpr,roc_auc])

def compute_AUC(C):
        for c in C:
            cv=StratifiedKFold(n_splits=10)

            mean_fpr=np.linspace(0,1,100)

            print("Start New mp with parameter C = "+str(c))
            pool=mp.Pool(processes=10)#, initializer=init)
            results = pool.map(cv.split(latent_pat,tag_mat))
            print("Results dim for C = "+str(c)+" is "+str(len(results)))
            pool.close()
            pool.join()

            tprs=[interp(mean_fpr,fpr,tpr) for fpr,tpr,roc_auc in results]
            roc_aucs=[auc(fpr,tpr) for fpr,tpr,roc_auc in results]

            plt.figure()
            for fpr,tpr,roc_auc in results:
                plt.plot(fpr,tpr,lw=1,alpha=0.3)

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(roc_aucs)
            plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
            print("Mean AUC for C="+str(c)+" is "+str(mean_auc))

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            print("Saving Figure for C = "+str(c))
            plt.savefig(file_path+"AUC_SVM_C"+str(c).replace(".","_")+".pdf")
            print("Done with thread C = "+str(c))
        return(0)

compute_AUC([0])
