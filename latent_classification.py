#Death classification
import torch
import pandas as pd
import numpy as np
from sklearn import svm
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

from sklearn import datasets

import sys

from functools import partial

file_path=sys.argv[1:][0]

if "macau" in file_path:
    latent_pat=np.load(file_path+"mean_pat_latent.npy").T
else:
    latent_pat=torch.load(file_path+"best_model.pt")["pat_lat.weight"].cpu().numpy() #latents without covariates
#print(latent_pat.shape)
    covariates=pd.read_csv("~/Data/MIMIC/complete_covariates.csv").as_matrix() #covariates
    beta_u=torch.load(file_path+"best_model.pt")["beta_u"].cpu().numpy() #Coeffs for covariates
    latent_pat_cov=np.dot(covariates[:,1:],beta_u)
    print("SHAPES")
    print(latent_pat.shape)
    print(latent_pat_cov.shape)

tags=pd.read_csv("~/Data/MIMIC/complete_death_tags.csv").sort_values("UNIQUE_ID")
tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]
print(tag_mat.shape)
print(latent_pat.shape)

#testdata
#iris=datasets.load_iris()
#X=iris.data
#y=iris.target%2
#latent_pat=X
#tag_mat=y


print("Data is Loaded")

def roc_comp(train_test,c,gamma):
    #print("roc_comp with C parameter ="+str(clf.get_params()["C"]))
    clf=svm.SVC(C=c,class_weight="balanced",probability=True,kernel="rbf",gamma=gamma)
    probas_=clf.fit(latent_pat[train_test[0]],tag_mat[train_test[0]]).predict_proba(latent_pat[train_test[1]])
    fpr, tpr, thresholds = roc_curve(tag_mat[train_test[1]], probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    return([fpr,tpr,roc_auc])

def init():
    global latent_pat, tag_mat
    latent_pat=torch.load(file_path+"current_model.pt")["pat_lat.weight"].numpy()
    tags=pd.read_csv("~/Data/MIMIC/death_tag_tensor.csv").sort_values("UNIQUE_ID")
    tag_mat=tags[["DEATHTAG","UNIQUE_ID"]].as_matrix()[:,0]

    print("initialization complete")

def compute_AUC(C,gamma):

        for c in C:
            for g in gamma:
                cv=StratifiedKFold(n_splits=10)
                #print("Baseline : "+str(1-np.sum(tag_mat)/tag_mat.shape[0]))

                #global clf
                #clf=svm.SVC(C=c,class_weight="balanced",probability=True,kernel="linear")


                mean_fpr=np.linspace(0,1,100)

                print("Start New mp with parameter C = "+str(c))
                pool=mp.Pool(processes=10)#, initializer=init)
                results = pool.map(partial(roc_comp,c=c,gamma=g),cv.split(latent_pat,tag_mat))
                print("Results dim for C = "+str(c)+" is "+str(len(results)))
                pool.close()
                pool.join()

                tprs=[interp(mean_fpr,fpr,tpr) for fpr,tpr,roc_auc in results]
                roc_aucs=[auc(fpr,tpr) for fpr,tpr,roc_auc in results]

                plt.figure()
                for fpr,tpr,roc_auc in results:
                    plt.plot(fpr,tpr,lw=1,alpha=0.3)
                #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

                    #tprs.append(interp(mean_fpr, fpr, tpr))
                    #tprs[-1][0] = 0.0
                    #roc_auc = auc(fpr, tpr)
                    #aucs.append(roc_auc)
                    #plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
                    #i+=1

                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(roc_aucs)
                plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
                print("Mean AUC for C="+str(c)+" and gamma="+str(g)+" is "+str(mean_auc))

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
    #return([mean_fpr,mean_tpr,std_tpr,c,mean_auc,std_auc])

class NoDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self,value):
        pass
    daemon = property(_get_daemon,_set_daemon)

class MyPool(PoolParent):
    Process=NoDaemonProcess


C_vec=[0.1,1,10,100]
gamma_Vec=[0.0001,0.01,1]
#main_pool=MyPool(processes=3)
#res=[main_pool.apply_async(compute_AUC,(c,)) for c in C_vec]
#result_fin=[r.get() for r in res]
compute_AUC(C_vec,gamma_Vec)
print("Processing Finished")
# for res_vec in result_fin:
#     plt.figure()
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)
#     mean_fpr=res_vec[0]
#     mean_tpr=res_vec[1]
#     std_tpr=res_vec[2]
#     c=res_vec[3]
#     mean_auc=res_vec[4]
#     std_auc=res_vec[5]
#     print(mean_fpr.shape)
#
#     plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
#
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
#
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     print("Saving Figure for C = "+str(c))
#     plt.savefig(file_path+"AUC_SVM_C"+str(c).replace(".","_")+".pdf")
#     print("Done with thread C = "+str(c))

    #scores=cross_val_score(clf,latent_pat,tag_mat,cv=10)
    #print("C values : "+str(c))
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #mean_res=mean_res+scores.mean()
    #std_res=std_res+scores.std()
