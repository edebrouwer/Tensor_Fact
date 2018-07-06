
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tensor_utils import tensor_fact,TensorFactDataset, TensorFactDataset_ByPat

import os
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

    N_t=97 # NUmber of time steps
    #With Adam optimizer
    opt=parser.parse_args()
    str_dir="./trained_models/"
    for key in vars(opt):
        str_dir+=str(key)+str(vars(opt)[key])+"_"
    str_dir+="/"

    #Check if output directory exits, otherwise, create it.
    if (not os.path.exists(str_dir)):
        os.makedirs(str_dir)
    else:
        replace_prev=input("This configuration has already been run !Do you want to continue ? y/n")
        if (replace_prev=="n"):
            raise ValueError("Aborted")
    import time

    #Gpu selection
    gpu_id="1"
    if opt.gpu_name=="Tesla":
        gpu_id="0"

    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

    if opt.cuda:
        device=torch.device("cuda:0")
    else:
        device=torch.device("cpu")
    print("Device : "+str(device))

    print("GPU num used : "+str(torch.cuda.current_device()))
    print("GPU used : "+str(torch.cuda.get_device_name(torch.cuda.current_device())))

    if opt.hard_split:
        suffix="_HARD.csv"
    else:
        suffix=".csv"


    train_hist=np.array([])
    val_hist=np.array([])

    if opt.by_pat:
        if opt.DL:
            raise ValueError("Deep Learning with data feeding by patient is not supported yet")
        elif opt.death_label:
            train_dataset=TensorFactDataset_ByPat(csv_file_serie="lab_short_tensor.csv") #Full dataset for the Training
        else:
            train_dataset=TensorFactDataset_ByPat(csv_file_serie="lab_short_tensor_train"+suffix)
            val_dataset=TensorFactDataset_ByPat(csv_file_serie="lab_short_tensor_val"+suffix)

        mod=tensor_fact(device=device,covariates=train_dataset.cov_u,mode="by_pat",n_pat=train_dataset.length,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
        mod.double()
        mod.to(device)

        fwd_fun=mod.forward_full
    else:
        if opt.DL:
            train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train"+suffix)
            val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val"+suffix)
            mod=tensor_fact(device=device,covariates=train_dataset.cov_u,mode="Deep",n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
            fwd_fun=mod.forward_DL
        elif opt.death_label:
            train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor.csv") #Full dataset for the Training
            mod=tensor_fact(device=device,covariates=train_dataset.cov_u,mode="Class",n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
            fwd_fun=mod.forward
        elif opt.XT:
            train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train"+suffix)
            val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val"+suffix)
            mod=tensor_fact(device=device,covariates=train_dataset.cov_u,mode="XT",n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
            fwd_fun=mod.forward_XT
        else:
            train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train"+suffix)
            val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val"+suffix)
            mod=tensor_fact(device=device,covariates=train_dataset.cov_u,mode="Normal",n_pat=train_dataset.pat_num,n_meas=train_dataset.meas_num,n_t=N_t,l_dim=opt.latents,n_u=train_dataset.covu_num,n_w=1)
            mod.double()
            mod.to(device)
            fwd_fun=mod.forward

    dataloader = DataLoader(train_dataset, batch_size=opt.batch,shuffle=True,num_workers=2)
    if not opt.death_label:
        dataloader_val = DataLoader(val_dataset, batch_size=len(val_dataset),shuffle=False)

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
                preds=fwd_fun(indexes)
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
                preds=fwd_fun(indexes[:,0],indexes[:,1],indexes[:,2])
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
                        pred_val=fwd_fun(indexes)
                        pred_val=torch.masked_select(pred_val,mask)
                    else:
                        indexes=batch_val[:,1:4].to(torch.long).to(device)
                        target=batch_val[:,-1].to(device)
                        pred_val=fwd_fun(indexes[:,0],indexes[:,1],indexes[:,2])
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
