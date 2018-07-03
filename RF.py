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

import optunity
import optunity.metrics

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
    clf=RandomForestClassifier(max_depth=None,n_estimators=2000,class_weight="balanced")
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
            results = pool.map(roc_comp,cv.split(latent_pat,tag_mat))
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
            plt.savefig(file_path+"AUC_RF_C"+str(c).replace(".","_")+".pdf")
            print("Done with thread C = "+str(c))
        return(0)

@optunity.cross_validated(x=latent_pat,y=tag_mat,num_folds=5)
def RF_auc(x_train,y_train,x_test,y_test,log_n_est,max_depth):
        model=RandomForestClassifier(max_depth=int(max_depth),n_estimators=int(10**log_n_est),class_weight="balanced")
        probs_=model.fit(x_train, y_train).predict_proba(x_test)
        auc_roc=optunity.metrics.roc_auc(y_test,probs_[:,1])
        print("AUC for n_est ="+str(10**log_n_est)+"and max_depth ="+str(max_depth)+" is "+str(auc_roc))
        return(auc_roc)

hps, a, b = optunity.maximize(RF_auc,num_evals=200,log_n_est=[1.5,3.5],max_depth=[1,30])
optimal_model=RandomForestClassifier(n_estimators=int(10**hps['log_n_est']),max_depth=hps['max_depth'],class_weight="balanced")
print(hps["n_estimators"])
#compute_AUC([0])
#optunity_RF()
