class tensor_fact(nn.Module):
    def __init__(self,device,covariates,n_pat=10,n_meas=5,n_t=25,l_dim=2,n_u=2,n_w=3,mode="Normal",l_kernel=3,sig2_kernel=1):
        #mode can be Normal, Deep, XT , Class or  by_pat
        super(tensor_fact,self).__init__()
        self.n_pat=n_pat
        self.n_meas=n_meas
        self.n_t=n_t
        self.l_dim=l_dim
        self.n_u=n_u
        self.n_w=n_w
        self.pat_lat=nn.Embedding(n_pat,l_dim) #sparse gradients ?
        self.pat_lat.weight=nn.Parameter(0.05*torch.randn([n_pat,l_dim]))
        self.meas_lat=nn.Embedding(n_meas,l_dim)
        self.meas_lat.weight=nn.Parameter(0.05*torch.randn([n_meas,l_dim]))
        self.time_lat=nn.Embedding(n_t,l_dim)#.double()
        self.time_lat.weight=nn.Parameter(0.005*torch.randn([n_t,l_dim]))
        self.beta_u=nn.Parameter(torch.randn([n_u,l_dim],requires_grad=True))#.double())
        self.beta_w=nn.Parameter(torch.randn([n_w,l_dim],requires_grad=True))#.double())

        self.cov_w_fixed=torch.tensor(range(0,n_t),device=device,dtype=torch.double,requires_grad=False).unsqueeze(1)#.double()
        self.covariates_u=covariates.to(device)
        #self.covariates_u.load_state_dict({'weight':covariates})
        #self.covariates_u.weight.requires_grad=False

        full_dim=3*l_dim+n_u+n_w
        #print(full_dim)

        if (mode=="Deep"):
            self.layer_1=nn.Linear(full_dim,50)
            self.layer_2=nn.Linear(50,50)
            self.layer_3=nn.Linear(50,20)
            self.last_layer=nn.Linear(20,1)

        if (mode=="Class"):
            #classification
            self.layer_class_1=nn.Linear((l_dim+n_u),20)
            self.layer_class_2=nn.Linear(20,1)

        if (mode=="XT"):
            self.layer_1=nn.Linear(l_dim,20)
            self.layer_2=nn.Linear(20,20)
            self.layer_3=nn.Linear(20,1)

        #Kernel_computation
        #x_samp=np.linspace(0,(n_t-1),n_t)
        #SexpKernel=np.exp(-(np.array([x_samp]*n_t)-np.expand_dims(x_samp.T,axis=1))**2/(2*l_kernel**2))
        #SexpKernel[SexpKernel<0.1]=0
        #self.inv_Kernel=torch.tensor(np.linalg.inv(SexpKernel)/sig2_kernel,requires_grad=False)

    def forward(self,idx_pat,idx_meas,idx_t,cov_u,cov_w):
        pred=((self.pat_lat(idx_pat)+torch.mm(self.covariates_u[idx_pat,:],self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(cov_w,self.beta_w))).sum(1)
        return(pred)
    def forward_full(self,idx_pat,cov_u):
        #cov_w=torch.tensor(range(0,101)).unsqueeze(1)#.double()
        pred=torch.einsum('il,jkl->ijk',((self.pat_lat(idx_pat)+torch.mm(cov_u,self.beta_u),torch.einsum("il,jl->ijl",(self.meas_lat.weight,(self.time_lat.weight+torch.mm(self.cov_w_fixed,self.beta_w)))))))
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
    def forward_XT(self,idx_pat,idx_meas,idx_t,cov_u,cov_w):
        latent=((self.pat_lat(idx_pat)+torch.mm(self.covariates_u[idx_pat,:],self.beta_u))*(self.meas_lat(idx_meas))*(self.time_lat(idx_t)+torch.mm(cov_w,self.beta_w)))
        out=F.relu(self.layer_1(latent))
        out=F.relu(self.layer_2(out))
        out=F.relu(self.layer_3(out)).squeeze(1)
        return(out)

    def label_pred(self,idx_pat,cov_u): #Classifiction task
        merged_input=torch.cat((self.pat_lat(idx_pat),cov_u),1)
        out=F.relu(self.layer_class_1(merged_input))
        out=F.sigmoid(self.layer_class_2(out))
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

        #Randomly select patients for classification validation.
        self.test_idx=np.random.choice(self.pat_num,size=int(0.2*self.pat_num),replace=False) #0.2 validation rate
        self.lab_short.loc[self.lab_short["UNIQUE_ID"].isin(self.test_idx),"DEATHTAG"]=np.nan
        self.tensor_mat=self.lab_short.as_matrix()

        self.tags=pd.read_csv(file_path+"death_tag_tensor.csv")
        self.test_labels=torch.tensor(self.tags.loc[self.tags["UNIQUE_ID"].isin(self.test_idx)].sort_values(by="UNIQUE_ID").as_matrix())
       # self.test_covariates=torch.tensor(self.lab_short.loc[self.lab_short["UNIQUE_ID"].isin(self.test_idx)].sort_values(by="UNIQUE_ID")[self.cov_values].as_matrix()).to(torch.double)
        covariates=self.lab_short.groupby("UNIQUE_ID").first()[self.cov_values].reset_index()
        self.test_covariates=torch.tensor(covariates.loc[covariates["UNIQUE_ID"].isin(self.test_idx)].sort_values(by="UNIQUE_ID")[self.cov_values].as_matrix()).to(torch.double)
        print(self.test_covariates.size())
        print(self.test_labels.size())

        self.cov_u=torch.tensor(pd.read_csv(file_path+"lab_covariates_val.csv").as_matrix()[:,1:]).to(torch.double)

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
        #covariates=self.lab_short.groupby("UNIQUE_ID").first()[cov_values]
        #self.cov_u=torch.DoubleTensor(covariates.as_matrix())
        self.cov_u=torch.tensor(pd.read_csv(file_path+"lab_covariates_val.csv").as_matrix()[:,1:]).to(torch.double)
        self.length=self.cov_u.size(0)
       # print(self.cov_u.size())
       # print(self.data_matrix.size())
        self.tags=pd.read_csv(file_path+"death_tag_tensor.csv").as_matrix()[:,1]
        self.test_idx=np.random.choice(self.length,size=int(0.2*self.length),replace=False) #0.2 validation rate
        self.train_tags=self.tags
        self.train_tags[self.test_idx]=np.nan

    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return([idx,self.data_matrix[idx,:,:],self.cov_u[idx,:],self.train_tags[idx]])
