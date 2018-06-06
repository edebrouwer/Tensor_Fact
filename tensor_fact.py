
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
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
#GPU args
parser.add_argument('--cuda',action='store_true')
parser.add_argument('--gpu_name',default='Titan',type=str,help="Name of the gpu to use for computation")
#Savings args
parser.add_argument('--outfile',default="./",type=str,help="Path to save the models and outpus")


class tensor_fact(nn.Module):
    def __init__(self,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3,l_kernel=3,sig2_kernel=1):
        super(tensor_fact,self).__init__()
        self.n_pat=n_pat
        self.n_meas=n_meas
        self.n_t=n_t
        self.l_dim=l_dim
        self.n_u=n_u
        self.n_w=n_w
        self.pat_lat=nn.Embedding(n_pat,l_dim) #sparse gradients ?
        self.pat_lat.weight=nn.Parameter(0.25*torch.randn([n_pat,l_dim]))
        self.meas_lat=nn.Embedding(n_meas,l_dim)
        self.meas_lat.weight=nn.Parameter(0.25*torch.randn([n_meas,l_dim]))
        self.time_lat=nn.Embedding(n_t,l_dim)#.double()
        self.time_lat.weight=nn.Parameter(0.005*torch.randn([n_t,l_dim]))
        self.beta_u=nn.Parameter(torch.randn([n_u,l_dim],requires_grad=True))#.double())
        self.beta_w=nn.Parameter(torch.randn([n_w,l_dim],requires_grad=True))#.double())

        full_dim=3*l_dim+n_u+n_w
        #print(full_dim)
        self.layer_1=nn.Linear(full_dim,50)
        self.layer_2=nn.Linear(50,50)
        self.layer_3=nn.Linear(50,20)
        self.last_layer=nn.Linear(20,1)

        #Kernel_computation
        x_samp=np.linspace(0,(n_t-1),n_t)
        SexpKernel=np.exp(-(np.array([x_samp]*n_t)-np.expand_dims(x_samp.T,axis=1))**2/(2*l_kernel**2))
        SexpKernel[SexpKernel<0.1]=0
        self.inv_Kernel=torch.tensor(np.linalg.inv(SexpKernel)/sig2_kernel,requires_grad=False)

    def forward(self,idx_pat,idx_meas,idx_t,cov_u,cov_w):
        pred=((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)
    def forward_full(self,idx_pat,cov_u):
        cov_w=torch.tensor(range(0,101)).unsqueeze(1).double()
        pred=torch.einsum('il,jkl->ijk',((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u),torch.einsum("il,jl->ijl",(self.meas_lat.weight,(self.time_lat.weight+torch.mm(cov_w,self.beta_w)))))))
        #pred=((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u))*(self.meas_lat.weight)*(self.time_lat.weight+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)
    def forward_DL(self,idx_pat,idx_meas,idx_t,cov_u,cov_w):
        #print("Type of patlat "+str(self.pat_lat(idx_pat).type()))
        #print("Type of patlat "+str(self.meas_lat(idx_meas).type()))
        #print("Type of patlat "+str(self.time_lat(idx_t).type()))
        #print("Type of patlat "+str(cov_u.type()))
        #print("Type of patlat "+str(cov_w.type()))
        #merged_input=torch.cat((self.pat_lat(idx_pat),self.meas_lat(idx_meas),self.time_lat(idx_t),cov_u,cov_w),1)
        merged_input=torch.cat((self.pat_lat(idx_pat),self.meas_lat(idx_meas),self.time_lat(idx_t),cov_u,cov_w),1)
        #print(merged_input.size())
        out=F.relu(self.layer_1(merged_input))
        out=F.relu(self.layer_2(out))
        out=F.relu(self.layer_3(out))
        out=self.last_layer(out).squeeze(1)
        return(out)
    def compute_regul(self):
        regul=torch.trace(torch.exp(-torch.mm(torch.mm(torch.t(self.time_lat.weight),self.inv_Kernel),self.time_lat.weight)))
        return(regul)



class TensorFactDataset(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor.csv",file_path="~/Data/MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.length=len(self.lab_short.index)
        self.pat_num=self.lab_short["UNIQUE_ID"].nunique()
        self.cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
        self.time_values=["TIME_STAMP","TIME_SQ"]
        self.tensor_mat=self.lab_short.as_matrix()
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return(self.tensor_mat[idx,:])

class TensorFactDataset_ByPat(Dataset):
    def __init__(self,csv_file_serie="lab_short_tensor.csv",file_path="~/Data/MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        idx_mat=self.lab_short[["UNIQUE_ID","LABEL_CODE","TIME_STAMP","VALUENORM"]].as_matrix()
        idx_tens=torch.LongTensor(idx_mat[:,:-1])
        val_tens=torch.DoubleTensor(idx_mat[:,-1])
        sparse_data=torch.sparse.DoubleTensor(idx_tens.t(),val_tens)
        self.data_matrix=sparse_data.to_dense()
        cov_values=[chr(i) for i in range(ord('A'),ord('A')+18)]
        covariates=self.lab_short.groupby("UNIQUE_ID").first()[cov_values]
        self.cov_u=torch.DoubleTensor(covariates.as_matrix())
        self.length=self.cov_u.size(0)
       # print(self.cov_u.size())
       # print(self.data_matrix.size())
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return([idx,self.data_matrix[idx,:,:],self.cov_u[idx,:]])

def main():
    #With Adam optimizer
    opt=parser.parse_args()

    #Check if output directory exits, otherwise, create it.
    if (not os.path.exists(opt.outfile)):
        os.makedirs(opt.outfile)

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



    train_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_train_HARD.csv")
    val_dataset=TensorFactDataset(csv_file_serie="lab_short_tensor_val_HARD.csv")

    train_hist=np.array([])
    val_hist=np.array([])

    mod=tensor_fact(n_pat=train_dataset.pat_num,n_meas=30,n_t=101,l_dim=opt.latents,n_u=18,n_w=1)
    mod.double()
    mod.to(device)

    if opt.DL:
        fwd_fun=mod.forward_DL
    else:
        if opt.by_pat:
            fwd_fun=mod.forward_full
            train_dataset=TensorFactDataset_ByPat(csv_file_serie="lab_short_tensor_train_HARD.csv")
            val_dataset=TensorFactDataset_ByPat(csv_file_serie="lab_short_tensor_val_HARD.csv")
        else:
            fwd_fun=mod.forward

    dataloader = DataLoader(train_dataset, batch_size=opt.batch,shuffle=True,num_workers=30)
    dataloader_val = DataLoader(val_dataset, batch_size=len(val_dataset),shuffle=False)

    optimizer=torch.optim.Adam(mod.parameters(), lr=opt.lr,weight_decay=opt.L2) #previously lr 0.03 with good rmse
    criterion = nn.MSELoss()#
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
                cov_u=sampled_batch[2].to(device)
                target=sampled_batch[1].to(device)
                mask=target.ne(0)
                target=masked_select(target,mask)
                optimizer.zero_grad()
                preds=fwd_fun(indexes,cov_u)
                preds=masked_select(preds,mask)
            else:
                indexes=sampled_batch[:,1:4].to(torch.long).to(device)
                #print("Type of index : "+str(indexes.dtype))
                cov_u=sampled_batch[:,4:22].to(device)
                cov_w=sampled_batch[:,3].unsqueeze(1).to(device)
                target=sampled_batch[:,-1].to(device)

                optimizer.zero_grad()
                preds=fwd_fun(indexes[:,0],indexes[:,1],indexes[:,2],cov_u,cov_w)
            #print(mod.compute_regul())
            loss=criterion(preds,target)#-mod.compute_regul()
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
            for i_val,batch_val in enumerate(dataloader_val):
                if opt.by_pat:
                    indexes=batch_val[0].to(torch.long).to(device)
                    cov_u=batch_val[2].to(device)
                    target=batch_val[1].to(device)
                    optimizer.zero_grad()
                    preds=fwd_fun(indexes,cov_u)
                else:
                    indexes=batch_val[:,1:4].to(torch.long).to(device)
                    cov_u=batch_val[:,4:22].to(device)
                    cov_w=batch_val[:,3].unsqueeze(1).to(device)
                    target=batch_val[:,-1].to(device)
                    pred_val=fwd_fun(indexes[:,0],indexes[:,1],indexes[:,2],cov_u,cov_w)
                loss_val=criterion(pred_val,target)
                print("Validation Loss :"+str(loss_val))
                val_hist=np.append(val_hist,loss_val)
                if loss_val<lowest_val:
                    torch.save(mod.state_dict(),opt.outfile+"best_model.pt")
                    lowest_val=loss_val

    torch.save(mod.state_dict(),opt.outfile+"current_model.pt")
    torch.save(train_hist,opt.outfile+"train_history.pt")
    torch.save(val_hist,opt.outfile+"validation_history.pt")

if __name__=="__main__":
    main()
