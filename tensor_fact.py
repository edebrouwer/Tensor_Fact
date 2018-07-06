
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tensor_utils import tensor_fact,TensorFactDataset, TensorFactDataset_ByPat, mod_select 

import os
import time
#Check Ethan Rosenthal github


parser=argparse.ArgumentParser(description="Longitudinal Tensor Factorization")

#Model parameters
parser.add_argument('--L2',default=0,type=float,help="L2 penalty (weight decay")
parser.add_argument('--lr',default=0.02,type=float,help="Learning rate of the optimizer")
parser.add_argument('--epochs',default=250,type=int,help="Number of epochs")
parser.add_argument('--latents',default=8,type=int,help="Number of latent dimensions")
parser.add_argument('--batch',default=65000,type=int,help="Number of samples per batch")

#Model selection
parser.add_argument('--DL',action='store_true',help="To switch to Deep Learning model")
parser.add_argument('--by_pat',action='store_true',help="To switch to by patient learning")
parser.add_argument('--death_label',action='store_true',help="Supervised training for the death labs")
parser.add_argument('--hard_split',action='store_true',help="To use the challenging validation splitting")
parser.add_argument('--XT',action='store_true',help="To use DL model on top of the product of latents")
#GPU args
parser.add_argument('--cuda',action='store_true')
parser.add_argument('--gpu_name',default='Titan',type=str,help="Name of the gpu to use for computation Titan or Tesla")
#Savings args
#parser.add_argument('--outfile',default="./",type=str,help="Path to save the models and outpus")


def main():

    opt=parser.parse_args()

    dataloader, dataloader_val, mod, device = mod_select(opt)

    train_hist=np.array([])
    val_hist=np.array([])

    optimizer=torch.optim.Adam(mod.parameters(), lr=opt.lr,weight_decay=opt.L2)
    criterion = nn.MSELoss()#
    class_criterion = nn.BCELoss()
    epochs_num=opt.epochs

    lowest_val=1
    for epoch in range(epochs_num):
        print("EPOCH : "+str(epoch))

        total_loss=0
        t_tot=0
        Epoch_time=time.time()
        for i_batch,sampled_batch in enumerate(dataloader):

            starttime=time.time()

            if opt.by_pat:
                indexes=sampled_batch[0].to(torch.long).to(device)
                #cov_u=sampled_batch[2].to(device)
                target=sampled_batch[1].to(device)
                mask=target.ne(0)
                target=torch.masked_select(target,mask)
                optimizer.zero_grad()
                preds=mod.forward(indexes)
                preds=torch.masked_select(preds,mask)
                if opt.death_label:
                    lab_target=sampled_batch[3].to(device)
                    lab_mask=(lab_target==lab_target)
                    lab_target=torch.masked_select(lab_target,lab_mask)
                    lab_preds=mod.label_pred(indexes,cov_u)
                    lab_preds=torch.masked_select(lab_preds,lab_mask)
            else:
                indexes=sampled_batch[:,1:4].to(torch.long).to(device)
                target=sampled_batch[:,-1].to(device)

                optimizer.zero_grad()
                preds=mod.forward(indexes[:,0],indexes[:,1],indexes[:,2])
                if opt.death_label:
                    lab_target=sampled_batch[:,4].to(torch.double).to(device)
                    lab_mask=(lab_target==lab_target)
                    lab_target=torch.masked_select(lab_target,lab_mask)
                    lab_preds=mod.label_pred(indexes[:,0],cov_u)
                    lab_preds=torch.masked_select(lab_preds,lab_mask.unsqueeze(1))
            #print(mod.compute_regul())
            loss=criterion(preds,target)#-mod.compute_regul()
            if opt.death_label:
                loss+=class_criterion(lab_preds,lab_target)
            loss.backward()
            # print(mod.time_lat.weight.grad)
            optimizer.step()
            t_flag=time.time()-starttime
            #print(t_flag)
            t_tot+=t_flag
            total_loss+=loss
        print("Current Training Loss :"+str(total_loss/(i_batch+1)))
        print("TOTAL TIME :"+str(time.time()-Epoch_time))
        print("Computation Time:"+str(t_tot))
        train_hist=np.append(train_hist,total_loss.item()/(i_batch+1))

        with torch.no_grad():
            if opt.death_label: #only classification loss
                if opt.by_pat:
                    val_preds=mod.label_pred(train_dataset.test_idx.to(device),train_dataset.test_cov_u[train_dataset.test_idx].to(device))
                    loss_val=class_criterion(val_preds,train_dataset.tags[train_dataset.test_idx].to(device))
                else:
                    val_preds=mod.label_pred(train_dataset.test_labels[:,3].to(torch.long).to(device),train_dataset.test_covariates.to(device))
                    loss_val=class_criterion(val_preds,train_dataset.test_labels[:,1].unsqueeze(1).to(device))

                print("Validation Loss :"+str(loss_val))
                val_hist=np.append(val_hist,loss_val)
                if loss_val<lowest_val:
                    torch.save(mod.state_dict(),str_dir+"best_model.pt")
                    lowest_val=loss_val
            else:
                for i_val,batch_val in enumerate(dataloader_val):
                    if opt.by_pat:
                        indexes=batch_val[0].to(torch.long).to(device)
                        #cov_u=batch_val[2].to(device)
                        target=batch_val[1].to(device)
                        mask=target.ne(0)
                        target=torch.masked_select(target,mask)
                        optimizer.zero_grad()
                        pred_val=mod.forward(indexes)
                        pred_val=torch.masked_select(pred_val,mask)
                    else:
                        indexes=batch_val[:,1:4].to(torch.long).to(device)
                        target=batch_val[:,-1].to(device)
                        pred_val=mod.forward(indexes[:,0],indexes[:,1],indexes[:,2])
                    loss_val=criterion(pred_val,target)
                    print("Validation Loss :"+str(loss_val))
                    val_hist=np.append(val_hist,loss_val)
                    if loss_val<lowest_val:
                        torch.save(mod.state_dict(),str_dir+"best_model.pt")
                        lowest_val=loss_val

    torch.save(mod.state_dict(),str_dir+"current_model.pt")
    torch.save(train_hist,str_dir+"train_history.pt")
    torch.save(val_hist,str_dir+"validation_history.pt")

    min_train=train_hist[np.where(val_hist==min(val_hist))]
    print("Training Error at lowest validation : "+str(min_train))
    print("Lowest Validation Error : "+str(min(val_hist)))

if __name__=="__main__":
    main()
