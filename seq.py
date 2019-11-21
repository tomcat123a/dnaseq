# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:00:22 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:56:40 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:52:46 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:25:58 2019

@author: Administrator
"""


import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,\
AdaptiveAvgPool1d,Linear,MSELoss,LSTM,GRU,MaxPool1d,AdaptiveMaxPool1d,AvgPool1d 
from torch.nn import  LeakyReLU,ReLU,Dropout
import torch.nn as nn
#from cus_activation import GELU
from os import path

import pandas as pd
import time
import numpy as np
from drnn import DRNN 
from parsedna import totensor
import os
from torch.utils.data import Dataset
import h5py
import math
from sklearn.ensemble import RandomForestRegressor
#generate sequence data
#generate expression value
#build models
#bayes optimization
#architecture search

#implement resnet,densenet

def conv3(in_channels, out_channels, stride=1 , dilation=1):
    """3x1 convolution with padding"""
    return Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=1, bias=False, dilation=dilation)


def conv1(in_channels, out_channels, stride=1,dilation=1,kernel=3):
    """1x1 convolution"""
    return Conv1d(in_channels, out_channels, kernel_size=kernel, dilation=dilation,padding=int((kernel-1)/2),stride=stride, bias=False)
    
class block0(torch.nn.Module):  #resnet
    expand=2
    def __init__(self,in_channels,out_channels,dilation,stride=1,first=False,res_on=True,kernel_size=3):
        #the first block will increase the channels even stride==1, or do downsampling and increasing channels 
        #even stride==1
        #if it is not the first block, then input x channels == out_channels, and then it will be shrunk to in_channels
        
        super(block0, self).__init__()
        inter_channels=min(int(out_channels/2),300)
        self.bn1=BatchNorm1d(out_channels)
        self.cv1=conv1(out_channels,inter_channels)
        self.ac1= ReLU()
        
        self.bn2=BatchNorm1d(inter_channels)
        self.ac2= ReLU()
        self.cv2=conv3(inter_channels,inter_channels,dilation=dilation,stride=1)
        self.bn3=BatchNorm1d(inter_channels)
        self.ac3= ReLU()
        self.cv3=conv1(inter_channels,out_channels)
        if first==True or stride!=1 or in_channels!=out_channels:
            self.downsample=downsample(in_channels,out_channels,stride,dilation=dilation,kernel=kernel_size)
        else:
            self.downsample=None
        self.res_on=res_on
    def forward(self, x ):
        #x.size() #N input_channels,C channels,L length
          #(N,C,L),C=2*in_channels
        if self.res_on:
            if self.downsample is not None:
                x0 = self.downsample(x)
            else:
                x0 = x
        else:
            x0=x
        x = self.bn1(x0)
        x = self.ac1(x)
        x = self.cv1(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.cv2(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.cv3(x)
        if self.res_on:
            x = x + x0
        #x = self.lin2(x)
        return x

class downsample(torch.nn.Module):  #resnet
     
    def __init__(self,in_channels,out_channels,stride,dilation,kernel):
        #N input_channels,C channels,L length
        super(downsample, self).__init__()
        
        #self.bn1=BatchNorm1d(in_channels)
         
        self.cv1=conv1(in_channels,out_channels,stride=stride,dilation=dilation,kernel=kernel)
        self.bn1=BatchNorm1d(in_channels)
        self.acti=ReLU()
         
    def forward(self, x ):
        x = self.bn1(x)
        x= self.cv1(x)
        
        x =self.acti(x)
        return x


    
class layer0(torch.nn.Module):  #resnet
    
    
    def __init__(self,in_channels,out_channels,block_type,depth,dilation,stride,**kwargs):
        #N input_channels,C channels,L length
        super(layer0, self).__init__()
        if not isinstance(dilation,list):
            di=[dilation]*depth
        else:
            if depth!=len(dilation):
                raise ValueError('depth!=len(dilation)')
            di=dilation
        self.block_list=[]
        for i in range(depth):
            if i==0:
                first=True
                b_stride=stride
            else:
                first=False
                b_stride=1
            self.block_list+=[block_type(in_channels=in_channels,out_channels=out_channels,dilation=di[i],stride=b_stride,first=first,**kwargs)]
        
        self.seq=Sequential(*self.block_list)
    def forward(self, x ):
        
        x = self.seq(x) 
        #x = self.lin2(x)
        return x
    


    
class block1(torch.nn.Module):  # multibranch
    expand=2
    def __init__(self,in_channels,out_channels,dilation,stride,splits,first=False,res_on=True):
        #N input_channels,C channels,L length
        super(block1, self).__init__()
        
        if first==False:
            self.bn1=BatchNorm1d(out_channels)
            #self.cv1=conv1(out_channels,in_channels)
        else:
            self.bn1=BatchNorm1d(in_channels)
            #self.cv1=conv1(in_channels,in_channels)
        self.ac1= ReLU()
        if  first==True or stride!=1:
            self.downsample=downsample(in_channels,out_channels,stride)
        else:
            self.downsample=None
        self.branch_list=[]
        self.splits=splits
        if first==True:
            for i in range(splits):
                self.branch_list = self.branch_list + [branch0(in_channels,out_channels,dilation,stride,splits)]
        else:
            for i in range(splits):
                self.branch_list = self.branch_list + [branch0(out_channels,out_channels,dilation,stride,splits)]
        self.res_on=res_on 
    def forward(self, x ):
        #print(x.size())
        if self.res_on:
            if self.downsample is not None:
                x0 = self.downsample(x)
                #print(x0.size())
            else:
                x0 = x
        x = self.bn1(x)
        x = self.ac1(x)
        x = torch.stack( [self.branch_list[i](x) for i in range(self.splits)] ).sum(0) 
        if self.res_on:
            x = x + x0
         
        #x = self.lin2(x)
        return x
    
class branch0(torch.nn.Module):  #densenet multibranch
    expand=2
    def __init__(self,in_channels,out_channels,dilation,stride,splits):
        #N in_channels,C channels,L length
        #input in_channels
        #output out_channels ,other dimension does not change
        super(branch0, self).__init__()
        if int(in_channels/splits)<1:
            raise ValueError('in_channels/splits should be at least 1 ')
        inter_channels=int(in_channels/splits)
        self.cv1=conv1(in_channels,inter_channels)
        self.bn2=BatchNorm1d(inter_channels)
        self.ac2= ReLU()
        self.cv2=conv3(inter_channels,inter_channels,dilation=dilation,stride=stride)
        self.bn3=BatchNorm1d(inter_channels)
        self.ac3= ReLU()
        self.cv3=conv1(inter_channels,out_channels)
         
    def forward(self, x ):
        
        x = self.cv1(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.cv2(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.cv3(x)
        
        #x = self.lin2(x)
        return x
    



def calc_num_par(x):
    #calculate the total number of parameters for a pytorch nn.Module
    pytorch_total_params = sum([p.numel() for p in x.parameters() if p.requires_grad])
    return  pytorch_total_params
 
class res(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_channels,n_layers,n_features,init_ker_size,block_type,depth,zero_init,degrid=True,tail=True,**kwargs):
        #N input_channels,C channels,L length
        super(res, self).__init__()
        if init_ker_size%2!=1:
            raise ValueError('init_ker_size must be an odd number')
        if 2**n_layers>n_features:
            raise ValueError('2**n_layers>n_features')
        self.layer_list=[]
        self.layer_list.append(Conv1d(in_channels, in_channels, kernel_size=init_ker_size, stride=1,
                     padding=int((init_ker_size-1)/2), groups=1, bias=False, dilation=1))
        factor=1.41
        for i in range(n_layers):
            
            self.layer_list.append(layer0(in_channels=int(factor**i*in_channels),out_channels=int(\
                                          factor**(i+1)*in_channels),block_type=block_type,stride=1,
                                          depth=depth,dilation=2**i,**kwargs))
            '''
            self.layer_list.append(layer0(in_channels=int(factor**i*in_channels),out_channels=int(\
                                          factor**(i+1)*in_channels),block_type=block_type,stride=2,
                                          depth=depth,dilation=1,**kwargs))
            '''
        if degrid == True:
            self.layer_list.append(layer0(in_channels=int(factor**n_layers*in_channels),out_channels=int(factor\
                                          **n_layers*in_channels),block_type=block_type,stride=1,
                                              depth=depth,dilation=1,**kwargs))
            
        self.seq=Sequential(*self.layer_list)
        self.avdpool=AdaptiveAvgPool1d(1)
        self.fc = Linear(int(factor**n_layers*in_channels), 1)
        for m in self.modules():
            if  isinstance(m,   BatchNorm1d ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init:
            for m in self.modules():
                if isinstance(m, block0):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, branch0):
                    nn.init.constant_(m.bn3.weight, 0)
        self.tail=tail
    def forward(self, x ):
        #print(x.size())
        x = self.seq(x)
        #print(x.size())
        if self.tail==True:
            x = self.avdpool(x)
            #print(x.size())
            x=x.view(x.size()[0],x.size()[1])
            x = self.fc(x)
        
        #x = self.lin2(x)
        return x
    
    
 

class biDRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False,rnn_gpu=False):
        super(biDRNN, self).__init__()
        self.drnn1=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type=cell_type, batch_first=False,rnn_gpu=rnn_gpu)
        self.drnn2=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type=cell_type, batch_first=False,rnn_gpu=rnn_gpu)
        
    def forward(self,input1):
        #input1 is (N,L,C) N batch size ,L seq length, C #channels
        input2 = torch.flip(input1,dims=[1])
        out1=self.drnn1(input1)[0]
        out2=self.drnn2(input2)[0]
         
        return torch.cat([out1,out2],dim=-1)



class testcnn(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_channels,n_layers,n_features,depth,block_type,degrid,testrnn,cell_type,hidden_size,n_layers_rnn,rnn_gpu,**kwargs):
        #N input_channels,C channels,L length
        super(testcnn, self).__init__()
        factor=1.41
        self.testrnn=testrnn
        self.cell_type=cell_type
        cnn_output_channels=int(in_channels*factor**(n_layers))
        cnn_output_len=int(self.shrink(n_features,n_layers))
        self.cnn_output_len = cnn_output_len
         
        self.prelayer=res(in_channels=in_channels,n_layers=n_layers,n_features=n_features,init_ker_size=7,\
              block_type=block_type,depth=depth,zero_init=True,degrid=degrid,tail=False,**kwargs)
        self.avdpool_cnn=AdaptiveAvgPool1d(1)
        #self.avdpool_cnn=Linear(cnn_output_len,1)
        #self.fc_cnn = Sequential(*[Linear(cnn_output_channels, cnn_output_channels),LeakyReLU(),Linear(cnn_output_channels, 1)])
        self.fc_cnn = Linear(cnn_output_channels, 1)
        
        if testrnn==True:
            if cell_type=='LSTM':
                self.rnn=LSTM(input_size=cnn_output_channels,hidden_size=hidden_size,num_layers=n_layers_rnn,bidirectional=True,batch_first=True)
            if cell_type=='GRU':
                self.rnn = GRU(input_size=cnn_output_channels,hidden_size=hidden_size,num_layers=n_layers_rnn,bidirectional=True,batch_first=True)
            if cell_type=='biDRNN':
                self.rnn = biDRNN(n_input=cnn_output_channels,n_hidden=hidden_size,n_layers=n_layers_rnn ,cell_type='GRU', batch_first=True,rnn_gpu= rnn_gpu)
            
            self.timedistributed_rnn=Linear(cnn_output_len,1)
            self.fc_rnn = Linear( 2*hidden_size, 1)
    def forward(self, x  ):
        x = self.prelayer(x)
         #N,C,L
        x1 = self.avdpool_cnn(x)
        x1 = x1.squeeze(dim=-1)
        #print(x1.size())
        x1 = self.fc_cnn(x1)
        
        if self.testrnn==True:
            #(N,C,L) to (N,L,C)
            x = x.transpose(1,2)
            #since rnn is batch first
            if self.cell_type!='biDRNN':
                x = self.rnn(x)[0] 
            else:
                x = self.rnn(x)
            x = x.transpose(1,2)
             
            x = self.timedistributed_rnn(x)
            x = x.squeeze(dim=-1)
            x = self.fc_rnn(x)
        #x = self.lin2(x)
            return x+x1
        else:
            return x1
    def shrink(self,n,m):
        x = n
        for i in range(m):
            if np.floor(x/2)==x/2:
                x = x/2
            else:
                x = np.floor(x/2)+1 
        return int(x)  
        
class SeqDataset(Dataset):
    """return both the training and test dataset."""

    def __init__(self,  tissue , exp_type , gen,chrom,pt_per,selected_patient_id,len_per,seq_folder='/home/yilun/dna/seqdata/',pack_size=32 ):
        """
        Args:
            tissue:values:0,1,2,3, ('Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood')
            exp_type:values: 0,1,2, 
            np (float) : percentage of number of patients selected.
            chrom in [6,8]
            len_per (float) :percentage of dna sequences selected
            n_train_gene_rate(float): ratio of the genes that are in the training data set, the 
            #rest are in the test data set.
            #seq folder '/home/yilun/dna/seqdata/chr8'
            #read seq data a = pd.read_csv('patient_id.txt',sep='\t') a.iloc[i,1] the first column at row i,
            #a.iloc[i,2] the second column at row i,a.iloc[:,0] is the gene-name list.
            #exp_folder '/home/yilun/dna/exp_nor_rbe/','NORM_RM.txt',
            #'/home/yilun/dna/exp_unnor_rbe/' ,'UNNORM_RM.txt', '/home/yilun/dna/exp_raw/','RAW.txt'
            #read expdata b = pd.read_csv('name.txt',sep='\t') b.iloc[i,0] is the expression value for
            first patient at gene i,
             b.index is the gene-name list,which is the same as a.iloc[:,0].b.columns is the list
             patient_id. The patients in this list will be fetched for the corresponding dna sequence in
             seq folder /home/yilun/dna/seqdata/SAMPLE2GENOTYPE.DICT
             
             Note:
                 1.all pd.read_csv must have sep='\t'
        """
        self.gen=gen
        if gen==False:
            #os.listdir() files for seq
            dicttable=pd.read_csv(seq_folder+'SAMPLE2GENOTYPE.DICT',sep='\t').values
            namedict=dict(zip( dicttable[:,0],dicttable[:,1]  ))
            if ( not (tissue in [0,1,2,3] ) ) or ( not (exp_type in [0,1,2]) ) or (not (pt_per<=1 and pt_per>=0)):
                raise ValueError( 'argument tissue in [0,1,2,3] and \
                                 exp_type in [0,1,2] and np<=1 and np>=0 does not hold!')
            TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood']
            EXP_FOLDER=['/home/yilun/dna/exp{}/exp_nor_rbe/'.format(chrom),'/home/yilun/dna/exp{}/exp_unnor_rbe/'.format(chrom),'/home/yilun/dna/exp{}/exp_raw/'.format(chrom)]
            EXP_TXT_NAME=['/NORM_RM.txt','/UNNORM_RM.txt','/RAW.txt']
            exp_folder_dir=EXP_FOLDER[exp_type]+TISSUE[tissue]+EXP_TXT_NAME[exp_type]
            exp_table=pd.read_csv(exp_folder_dir,sep='\t') 
             
            total_num_patients=exp_table.shape[1]
            if selected_patient_id is  None:
                selected_patient_id=np.random.choice(range(total_num_patients),int(total_num_patients*pt_per),replace=False)
            patient_list=np.take(list(exp_table.columns),selected_patient_id)
            #total_num_genes = exp_table.shape[0]
            
            #fetch the corresponding sequence
            self.train=[]
             
            for pt_id in patient_list:
                x =  pd.read_csv(seq_folder+'chr{}/'.format(chrom)+namedict[pt_id]+'.txt',sep='\t').iloc[:,1:]#two columns of strings
                 
                self.train.append(totensor(x,len_per)) #output size :#genes,4,seq_len
                 
            self.dfx=torch.cat(self.train,dim=0)#batch size(#genes*#patients),4,seq_len
            ##genes, #patients 
            self.dfy =  torch.from_numpy(exp_table.values.take(selected_patient_id,axis=1).reshape(-1,order='F').astype(np.float32)).type(torch.float32)
        else:
            n_seq = 40001
            n_channels = 4
            self.dfx=[]
            self.dfy=[]
            for i in range(10):
                x = np.zeros((  n_channels,n_seq ) )
                J = np.random.choice(n_channels, n_seq)
                x[J , np.arange(n_seq)] = 1
                self.dfx.append(x)
                self.dfy.append(sum(x[0:8,7])+0.5*sum(x[100:108,1])+\
                                2*sum(x[400:430,2])**2+0.05*sum(x[19900:200100,2:4])**2+0.1*np.random.rand(1))

# assign with advanced indexing
 
         

    def __len__(self):
        return len(self.dfy)

    def __getitem__(self, idx):
        if self.gen==False:
            return self.dfx[idx],self.dfy[idx]
        else:
            return torch.from_numpy(self.dfx[idx]).type(torch.float32),torch.from_numpy(self.dfy[idx]).type(torch.float32)

class SeqDatasetpt():
    """return both the training and test dataset."""

    def __init__(self,  tissue , exp_type , chrom,pt_per,selected_patient_id,len_per,seq_folder='/home/yilun/dna/seqdata/normalized/'  ,pack_size=32 ):
        """
        Args:
            tissue:values:0,1,2,3, ('Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood')
            exp_type:values: 0,1,2, 
            np (float) : percentage of number of patients selected.
            chrom in [6,8]
            len_per (float) :percentage of dna sequences selected
            n_train_gene_rate(float): ratio of the genes that are in the training data set, the 
            #rest are in the test data set.
            #seq folder '/home/yilun/dna/seqdata/chr8'
            #read seq data a = pd.read_csv('patient_id.txt',sep='\t') a.iloc[i,1] the first column at row i,
            #a.iloc[i,2] the second column at row i,a.iloc[:,0] is the gene-name list.
            #exp_folder '/home/yilun/dna/exp_nor_rbe/','NORM_RM.txt',
            #'/home/yilun/dna/exp_unnor_rbe/' ,'UNNORM_RM.txt', '/home/yilun/dna/exp_raw/','RAW.txt'
            #read expdata b = pd.read_csv('name.txt',sep='\t') b.iloc[i,0] is the expression value for
            first patient at gene i,
             b.index is the gene-name list,which is the same as a.iloc[:,0].b.columns is the list
             patient_id. The patients in this list will be fetched for the corresponding dna sequence in
             seq folder /home/yilun/dna/seqdata/SAMPLE2GENOTYPE.DICT
             
             Note:
                 1.all pd.read_csv must have sep='\t'
        """
        
        self.pack_size=pack_size
        #os.listdir() files for seq
        dicttable=pd.read_csv(seq_folder+'SAMPLE2GENOTYPE.DICT',sep='\t').values
        self.namedict=dict(zip( dicttable[:,0],dicttable[:,1]  ))
        if ( not (tissue in [0,1,2,3] ) ) or ( not (exp_type in [0,1,2]) ) or (not (pt_per<=1 and pt_per>=0)):
            raise ValueError( 'argument tissue in [0,1,2,3] and \
                             exp_type in [0,1,2] and np<=1 and np>=0 does not hold!')
        TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood']
        EXP_FOLDER=['/home/yilun/dna/exp{}/exp_nor_rbe/'.format(chrom),'/home/yilun/dna/exp{}/exp_unnor_rbe/'.format(chrom),'/home/yilun/dna/exp{}/exp_raw/'.format(chrom)]
        EXP_TXT_NAME=['/NORM_RM.txt','/UNNORM_RM.txt','/RAW.txt']
        exp_folder_dir=EXP_FOLDER[exp_type]+TISSUE[tissue]+EXP_TXT_NAME[exp_type]
        exp_table=pd.read_csv(exp_folder_dir,sep='\t') 
        self.total_num_genes=exp_table.shape[0]
        self.total_num_patients=exp_table.shape[1]
        if max(selected_patient_id)>=self.total_num_patients:
            raise ValueError('selected_patient_id can not exceed total number of patients,which is {}'.format(self.total_num_patients))
        if selected_patient_id is  None:
            selected_patient_id=np.random.choice(range(self.total_num_patients),int(self.total_num_patients*pt_per),replace=False)
        self.num_selected_patients=len(selected_patient_id)
        self.patient_list=np.take(list(exp_table.columns),selected_patient_id)
        #total_num_genes = exp_table.shape[0]
        
        #fetch the corresponding sequence
        self.fin=[] 
        self.fin.append(h5py.File("{}/{}_{}.h5".format(seq_folder+'chr{}pt/'.format(chrom), 'chr{}'.format(chrom), 0), "r"))
        self.fin.append( h5py.File("{}/{}_{}.h5".format(seq_folder+'chr{}pt/'.format(chrom), 'chr{}'.format(chrom), 1), "r"))
        self.fin.append( h5py.File("{}/{}_{}.h5".format(seq_folder+'chr{}pt/'.format(chrom), 'chr{}'.format(chrom), 2), "r")) 
        self.fin.append( h5py.File("{}/{}_{}.h5".format(seq_folder+'chr{}pt/'.format(chrom), 'chr{}'.format(chrom), 3), "r")) 
        self.fin.append( h5py.File("{}/{}_{}.h5".format(seq_folder+'chr{}pt/'.format(chrom), 'chr{}'.format(chrom), 4), "r")) 
        '''
        for pt_id in patient_list:
            #namedict[pt_id] patient genotype id
            gene_name=namedict[pt_id]
            if gene_name in fin0.keys() :
                self.train.append(fin0[gene_name]['S'])
             
            self.train.append(totensor(x,len_per)) #output size :#genes,4,seq_len
             
        self.dfx=torch.cat(self.train,dim=0)#batch size(#genes*#patients),4,seq_len
        ##genes, #patients 
        '''
        self.len_percent=len_per
        self.dfy =  torch.from_numpy(exp_table.values.take(selected_patient_id,axis=1).reshape(-1,order='F')).type(torch.float32)
        self.length=math.ceil(len(self.dfy)/self.pack_size/self.num_selected_patients)
        self.current_iter_idx=0
        self.shuffled_list = self.get_shuffled_list() 
        
# assign with advanced indexing
 
         
    def get_shuffled_list(self):
        
        '''
        neworder = np.random.choice(self.dfy.size()[0],self.dfy.size()[0],replace=False)
        out=[]
        for i in range(self.length):
            out.append(neworder[i*self.pack_size:(i+1)*self.pack_size])
        return out
        '''
        n = int(self.dfy.size()[0]/self.num_selected_patients) #full length
        neworder = list(range(n))
        ''' 
        for i in range(int(np.floor(n/self.pack_size/2))):
            neworder[i*self.pack_size*2:(i+1)*self.pack_size*2]=\
            np.random.choice(neworder[i*self.pack_size*2:(i+1)*self.pack_size*2],self.pack_size*2,replace=False)
        '''     
            
        if self.pack_size==1:
            raise ValueError('pack_size should be greater than 1')
        shift=np.random.randint(low=0,high=int(self.pack_size/2))
        out=[neworder[0:self.pack_size-shift]]
        for i in range(1,self.length):
            out.append(neworder[i*self.pack_size-shift:(i+1)*self.pack_size-shift])
        out[-1][-1]=n-1
        return out
    

    def get_pack(self, idx,pid=0):
        #pid represents which one of the patient_list is selected here,usually pateint_list has length 1,so pid has a default
        #value 0
        self.current_iter_idx = self.current_iter_idx + 1
        if idx >=self.length:
            raise ValueError('input argument of get_pack must be smaller than dataset length {}.'.format(self.length))
        if self.current_iter_idx==self.length-1:
            #new split
            self.current_iter_idx = 0
            
        #t0=time.time()     
        gene_indices=self.shuffled_list[idx] #gene index
          
        #patients index self.patient_list[quotient] patient names in terms of expression data
        
        patient_geno_name=self.namedict[self.patient_list[pid]]
        
        i=0
        while i < 5:
            if patient_geno_name in self.fin[i].keys():
                x1,y1= torch.from_numpy(self.fin[i][patient_geno_name]['S'][sorted(list(gene_indices))]).float(),\
            self.dfy[pid*self.total_num_genes+np.array(gene_indices)]
                #print('within get_pack it takes {}s'.format(time.time()-t0))
                #x1.size=batch,4,100001
                return x1[:,:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1],y1
            i=i+1

class Newloader(Dataset):
    """return both the training and test dataset."""

    def __init__(self,  tissue , exp_type ,chrom,section,pt_per,selected_patient_id,len_per,\
                 seq_folder='/media/yilun/Elements/DeepSeQ/unnormalized/',exp_folder='/media/yilun/Elements/DeepSeQ/expr/'\
,in_mem=False ):
        """
        Args:
            tissue:values:0,1,2,3, ('Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood')
            exp_type:values: 0,1,2, 
            np (float) : percentage of number of patients selected.
            chrom in [6,8]
            len_per (float) :percentage of dna sequences selected
            n_train_gene_rate(float): ratio of the genes that are in the training data set, the 
            #rest are in the test data set.
            #seq folder '/home/yilun/dna/seqdata/chr8'
            #read seq data a = pd.read_csv('patient_id.txt',sep='\t') a.iloc[i,1] the first column at row i,
            #a.iloc[i,2] the second column at row i,a.iloc[:,0] is the gene-name list.
            #exp_folder '/home/yilun/dna/exp_nor_rbe/','NORM_RM.txt',
            #'/home/yilun/dna/exp_unnor_rbe/' ,'UNNORM_RM.txt', '/home/yilun/dna/exp_raw/','RAW.txt'
            #read expdata b = pd.read_csv('name.txt',sep='\t') b.iloc[i,0] is the expression value for
            first patient at gene i,
             b.index is the gene-name list,which is the same as a.iloc[:,0].b.columns is the list
             patient_id. The patients in this list will be fetched for the corresponding dna sequence in
             seq folder /home/yilun/dna/seqdata/SAMPLE2GENOTYPE.DICT
             
             Note:
                 1.all pd.read_csv must have sep='\t'
        """
        #first columne of dicttable, is column names of exp_table
        #second columne of dicttable, is h5.keys()
        #os.listdir() files for seq
        self.len_percent=len_per
        self.section=section
        dicttable=pd.read_csv(exp_folder+'chr1/SAMPLE2GENOTYPE.DICT',sep='\t').values
        self.namedict=dict(zip( dicttable[:,0],dicttable[:,1]  ))
        if ( not (tissue in [0,1,2,3] ) ) or ( not (exp_type in [0,1,2]) ) or (not (pt_per<=1 and pt_per>=0)):
            raise ValueError( 'argument tissue in [0,1,2,3] and \
                             exp_type in [0,1,2] and np<=1 and np>=0 does not hold!')
            
        TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood',\
                'Adipose_Subcutaneous','Brain_Hippocampus','Esophagus_Mucosa','Prostate','Adipose_Visceral_Omentum','Brain_Hypothalamus'\
                ,'Skin_Not_Sun_Exposed_Suprapubic','Esophagus_Muscularis','Adrenal_Gland','Brain_Nucleus_accumbens_basal_ganglia','Fallopian_Tube',\
                'Skin_Sun_Exposed_Lower_leg','Artery_Aorta','Brain_Putamen_basal_ganglia','Heart_Atrial_Appendage','Small_Intestine_Terminal_Ileum',\
                'Artery_Coronary','Brain_Spinal_cord_cervical_c-1','Heart_Left_Ventricle','Spleen','Artery_Tibial','Brain_Substantia_nigra',\
                'Kidney_Cortex','Stomach','Bladder','Liver','Testis',\
                'Brain_Amygdala','Cells_EBV-transformed_lymphocytes','Lung','Thyroid','Brain_Anterior_cingulate_cortex_BA24','Cells_Transformed_fibroblasts',\
                'Minor_Salivary_Gland','Uterus','Brain_Caudate_basal_ganglia','Cervix_Ectocervix','Muscle_Skeletal','Vagina',\
                'Brain_Cerebellar_Hemisphere','Cervix_Endocervix','Nerve_Tibial','Brain_Cerebellum','Colon_Sigmoid','Brain_Cortex','Pancreas',\
                'Colon_Transverse','Brain_Frontal_Cortex_BA9','Esophagus_Gastroesophageal_Junction','Pituitary']
        
        
         
        
        EXP_TXT_NAME=['/NORM_RM.txt','/UNNORM_RM.txt','/RAW.txt']
        exp_folder_dir_list=[]
        for j in chrom:
            EXP_FOLDER=[exp_folder+'chr{}/exp_nor_rbe/'.format(j),exp_folder+'chr{}/exp_unnor_rbe/'.format(j),\
                    exp_folder+'chr{}/exp_raw/'.format(j)]
            exp_folder_dir_list.append(EXP_FOLDER[exp_type]+TISSUE[tissue]+EXP_TXT_NAME[exp_type])
         
        
        if os.name!='nt':
            exp_table=pd.concat([ pd.read_csv(exp_folder_dir_list[j],sep='\t') for j in range(len(chrom))])
        else:
            exp_table=pd.read_csv('C:/tmp/NORM_RM.txt',sep='\t') 
        #print(exp_table )
        
        self.train=[]
        self.h5 = []
        for i1 in chrom:
            for i2 in section:
                self.h5.append( h5py.File(seq_folder+'chr{}_{}.h5'.format(i1,i2), "r") )
        #print([ self.h5[k].keys()  for k in section ])
        whole_patient_list=[]
        for k in section:
            whole_patient_list+=self.h5[k].keys()  
         
        #len(self.h5)=len(chrom)*len(section)
        #filter exp_table to obtain expressions for patients within section by selecting columns of exp_table(axis=1)
        exp_table=exp_table.take([i  for i in range(len(exp_table.columns)) \
                                  if self.namedict[exp_table.columns[i]] \
                                  in whole_patient_list ],axis=1)
        #print(exp_table )
        total_num_patients=exp_table.shape[1]
        if selected_patient_id is  None:
            selected_patient_id=np.random.choice(range(total_num_patients),int(total_num_patients*pt_per),replace=False)
        
        
        self.patient_list=np.take(list(exp_table.columns),selected_patient_id)#patient name in tissues data
        exp_table=exp_table.take(selected_patient_id,axis=1)
        temp_1=list(self.h5[0].keys())[0]
        self.gene_keys=[]
        for k in range(len(chrom)):
            self.gene_keys+=( list(self.h5[k*len(section)][temp_1].keys())  )
        #print(self.gene_keys)
        #filtering genes
        exp_table=exp_table.take([i for i in range(exp_table.shape[0]) if exp_table.index[i] in self.gene_keys],axis=0)
        #print(exp_table)
        self.chr_list=[]
        for k in range(len(chrom)):
            #temp_2=list(self.h5[k*len(section)].keys())[0]
            self.chr_list+=list(k*np.ones(len(self.h5[k*len(section)][temp_1].keys()),dtype=int) )
        #self.chr_list=np.concatenate(self.chr_list)
        
        self.patient_sec_list=[]#which section are the elements of self.patient_list in
        for i in range(len(self.patient_list)):
            for j in range(len(section)):
                if self.namedict[self.patient_list[i]] in self.h5[j].keys():
                    self.patient_sec_list.append(j)
        # gene names among all chrom,shared by all patients
        #print(self.patient_sec_list)
        #print('exp_table.shape')
        print('num_genes :{},num_patients;{}'.format(exp_table.shape[0],exp_table.shape[1]))
        self.exp_shape=[exp_table.shape[0],exp_table.shape[1]]
        #print(self.patient_sec_list)
        self.total_num_genes = exp_table.shape[0]
        self.in_mem=in_mem
        #fetch the corresponding sequence
        
        ##genes, #patients 
        self.dfy =  torch.from_numpy( exp_table.values.reshape(-1,order='F').astype(np.float32) ).type(torch.float32)
        
        if in_mem==True:
            self.train=[]
            for idx in range(len(self.dfy)):
                patient_num=int(idx/len(self.gene_keys))
                 
                genenum=idx%len(self.gene_keys)
                chr_num = self.chr_list[genenum]
                #print('patient_num {}'.format(patient_num))
                sec_num=self.patient_sec_list[patient_num]
                #print(torch.from_numpy(self.h5[chr_num*len(self.section)+sec_num][self.namedict[self.patient_list[patient_num]]][self.gene_keys[genenum]][:][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float())
                self.train.append( torch.from_numpy(self.h5[chr_num*len(self.section)+sec_num][self.namedict[self.patient_list[patient_num]]][self.gene_keys[genenum]][:][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() ) #output size :#genes,4,seq_len
            
            #print(self.train[0].size())     
            self.dfx=torch.stack(self.train)#batch size(#genes*#patients),4,seq_len
            #print('init')
            #print( self.dfx.size())

# assign with advanced indexing
 
         

    def __len__(self):
        
        return len(self.dfy)

    def __getitem__(self, idx):
        if self.in_mem:
            #print('in loader get_item')
            #print(self.dfx[idx].size())
            return self.dfx[idx],self.dfy[idx]
        else:
            patient_num=int(idx/len(self.gene_keys))
             
            genenum=idx%len(self.gene_keys)
            chr_num = self.chr_list[genenum]
            #print('patient_num {}'.format(patient_num))
            sec_num=self.patient_sec_list[patient_num]
            return torch.from_numpy(self.h5[chr_num*len(self.section)+sec_num][self.namedict[self.patient_list[patient_num]]][self.gene_keys[genenum]][:][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float(),self.dfy[idx]
        
              
                 
class testmlp(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,n_features,ker_size ,stride,channels_list,dnn_list,di=True,res_on=False,poolsize=[500,100],use_rnn=False,extra=False ):
        #N input_channels,C channels,L length
        super(testmlp, self).__init__()
        if dnn_list[-1]!=1 :
            raise ValueError('dnn_list[-1]!=1')
         
        if not isinstance(stride,list):
            self.stride=[stride]*(len(channels_list)-1)
        else:
            self.stride=stride
         
        self.cnn_list=[]
        self.di=di
        self.ker_size=ker_size
        
        #idx = [i for i in range(n_features-1, -1, -1)]
        #self.idx = torch.cuda.LongTensor(idx) 
        #self.idx = idx
        for i in range(len(channels_list)-1):
            if res_on==False:
                if len(self.di) >1:
                    self.cnn_list.append(Conv1d(channels_list[i], channels_list[i+1], kernel_size=ker_size[i], stride=self.stride[i],
                             padding=int((ker_size[i]-1)/2), groups=1, bias=True, dilation=self.di[i]))
                else:
                    self.cnn_list.append(Conv1d(channels_list[i], channels_list[i+1], kernel_size=ker_size[i], stride=self.stride[i],
                             padding=int(( ker_size[i]-1)/2), groups=1, bias=True, dilation=1))
                self.cnn_list.append(BatchNorm1d(channels_list[i+1]))
                self.cnn_list.append(ReLU())
                if len(poolsize)>0:
                    self.cnn_list.append( AdaptiveAvgPool1d(poolsize[i]))
            else:
                if len(self.di)>1:
                    self.cnn_list.append(block0(in_channels=channels_list[i],\
                                                out_channels=channels_list[i+1],dilation=self.di[i],stride=self.stride[i],first=(i==0),\
                                                res_on=True, kernel_size= ker_size[i]))
                    if len(poolsize)>0:
                        self.cnn_list.append( AdaptiveAvgPool1d(poolsize[i]))
                else:
                    self.cnn_list.append(block0(in_channels=channels_list[i],\
                                                out_channels=channels_list[i+1],dilation=1,stride=self.stride[i],first=(i==0),\
                                                res_on=True, kernel_size=ker_size[i]))
                    if len(poolsize)>0:
                        self.cnn_list.append( AdaptiveAvgPool1d(poolsize[i]))
        self.cnn_list=Sequential(*self.cnn_list) 
        cnn_out_length=self.shrink(n_features,len(channels_list)-1)
         
        #self.avdpool_cnn=Sequential(MaxPool1d(kernel_size=self.init_ker_size,padding=int((init_ker_size-1)/2), stride=self.stride ),
        if len(poolsize)>0:
            self.avdpool_cnn=Linear(poolsize[-1],1)
        
        #self.mlp_list.append(Linear(channels_list[-1],dnn_list[0]))
        #self.mlp_list.append(BatchNorm1d(dnn_list[0]))
        #self.mlp_list.append(LeakyReLU())
         
        ####rnn
        self.use_rnn=use_rnn
        self.mlp_list=[]
        self.extra=extra
        if self.use_rnn:
            hidden_size=10
            self.rnn = biDRNN(n_input=channels_list[-1],n_hidden=hidden_size,n_layers=2 ,cell_type='GRU', batch_first=True,rnn_gpu= True)
            #self.timedistributed_rnn=Linear(poolsize[-1],1)
            #self.fc_rnn = Linear( 2*hidden_size, 1)    
            self.mlp_list.append(Linear(poolsize[-1]*(channels_list[-1]+2*hidden_size),dnn_list[0]))
        else:
            if len(poolsize)>0:
                if extra==False:
                    self.mlp_list.append(Linear(poolsize[-1]*channels_list[-1],dnn_list[0]))
                else:
                    self.mlp_list.append(Linear(poolsize[-1]*channels_list[-1]+1,dnn_list[0]))
            else:
                if extra==False:
                    self.mlp_list.append(Linear(cnn_out_length*channels_list[-1],dnn_list[0]))
                else:
                    self.mlp_list.append(Linear(cnn_out_length*channels_list[-1]+1,dnn_list[0]))
        for i in range(len(dnn_list)-1):
            self.mlp_list.append(Linear(dnn_list[i],dnn_list[i+1]))
            if i< len(dnn_list)-2:
                self.mlp_list.append(BatchNorm1d(dnn_list[i+1]))
                self.mlp_list.append(ReLU())
                self.mlp_list.append(Dropout())
        if extra==True:
            self.embed_layer=embed([1])
        self.mlp_list=Sequential(*self.mlp_list)
    def forward(self, x  ):
        #x.size()=N,C,L（num_features）
        #print(x.size())
         
        x=x.contiguous()
        if self.extra:
            embed_x=self.embed_layer(x)
        x = self.cnn_list(x)#N,C,L
        #print(' x size')
        #print( x.size())
        if self.use_rnn==True:
            #rnn start
            z = x.transpose(1,2)#since rnn is batch first
            z = self.rnn(z)
            z = z.transpose(1,2)
            #print('z size')
            #print(z.size())
            x=torch.cat((x,z),dim=1)
        cnn_x = x.view(x.size()[0],-1).contiguous()
         
        
        if self.extra:
            cnn_x=torch.cat((cnn_x,embed_x),1)
        cnn_x = self.mlp_list(cnn_x)
        
        
        return cnn_x
        
        
    def shrink(self,n,m):
        x = n
        for i in range(m):
            ''' 
            if self.di:
                x = int((x+2*int((self.init_ker_size-1)/2)-2**i*(self.init_ker_size-1)-1)/self.stride[i]+1)
                
            else:
            ''' 
             
            x = int( (x+2*int((self.ker_size[i]-1)/2)-self.di[i]*(self.ker_size[i]-1)-1)/self.stride[i]+1)
             
        return x           

             
class testmlp_chunk(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,n_features,init_ker_size,stride,channels_list,dnn_list,di=True,\
                 res_on=False,poolsize=[500,100],use_rnn=False ,chunk_size=1,chunk_id=None):
        #N input_channels,C channels,L length
        super(testmlp_chunk, self).__init__()
        self.chunk_len=int(n_features/chunk_size)
        self.chunk_id=chunk_id
        if self.chunk_id is None:
            self.chunk_id=range(chunk_size)
        else:
            if not isinstance( self.chunk_id,list):
                self.chunk_id=[int(self.chunk_id)]
                
        self.mod=torch.nn.ModuleList()
        for j in range(1,chunk_size+1):
            self.mod.append( testmlp( n_features=int(n_features/chunk_size), init_ker_size=init_ker_size, \
            stride=stride,channels_list=channels_list, dnn_list=dnn_list, di=di, \
            res_on=res_on, poolsize=poolsize,use_rnn=use_rnn) )
        self.x_list=[]
        self.summarize=Sequential(Linear(len(self.chunk_id),3),ReLU(),BatchNorm1d(3),Linear(3,1))
    def forward(self, x  ):
        #x.size()=N,C,L（num_features）
        #print(x.size())
         
        x_tuple=torch.split(x,self.chunk_len,dim=-1)
        
        for j in self.chunk_id:
            if j==self.chunk_id[0]:
                x=self.mod[j](x_tuple[j])
            else:
                x = torch.cat((x,self.mod[j](x_tuple[j]) ),dim=1)
        #print(' x size')
        x = self.summarize(x)
        
        
        return x
        
        
           


class testlinear(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,n_features,dnn_list):
        #N input_channels,C channels,L length
        super(testlinear, self).__init__()
        if dnn_list[-1]!=1 :
            raise ValueError('dnn_list[-1]!=1 or channels_list[0]!=4')
        self.last=[Linear(4*n_features,dnn_list[0])]
        
        for i in range(len(dnn_list)-1):
            
            self.last.append(BatchNorm1d(dnn_list[i]))
            self.last.append(ReLU())
            self.last.append(Dropout())
            #self.last.append(Dropout())
            self.last.append(Linear(dnn_list[i],dnn_list[i+1]))
            #if i < len(dnn_list)-2:
                
         
        self.last=Sequential(*self.last) 
        '''
        for p in self.last.parameters():
            torch.nn.init.constant_(p, 0)
        '''
    def forward(self, x  ):
        #print(x.size())
        x = x.view(x.size()[0],-1)
        x = self.last(x)
         
        return x



class embed(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,dnn_list):
        #N input_channels,C channels,L length
        super(embed, self).__init__()
         
        self.last=[Linear(1,dnn_list[0])]
        
        for i in range(len(dnn_list)-1):
            
            self.last.append(BatchNorm1d(dnn_list[i]))
            self.last.append(ReLU())
            self.last.append(Dropout())
            #self.last.append(Dropout())
            self.last.append(Linear(dnn_list[i],dnn_list[i+1]))
            #if i < len(dnn_list)-2:
                
        self.layer0=testrna()
        self.last=Sequential(*self.last) 
        '''
        for p in self.last.parameters():
            torch.nn.init.constant_(p, 0)
        '''
    def forward(self, x  ):
        #print(x.size())
        x = self.layer0(x)
        x = x.view(x.size()[0],1)
        #print(x.size())
        x = self.last(x)
         
        return x    
    
    
class testrna(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self):
        #N input_channels,C channels,L length
        super(testrna, self).__init__()
        
            
        '''
        for p in self.last.parameters():
            torch.nn.init.constant_(p, 0)
        '''
    def forward(self, x  ):
        mid=x.size()[-1]
        z=( x[:,2,mid-2000:mid+14998].sum(1)+x[:,3,mid-2000:mid+14998].sum(1) )/x.size()[2]
         
        return z
     
class test_rnn(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,n_features,out_channels,hidden,n_layers,width,dnn_list):
        #N input_channels,C channels,L length
        super(test_rnn, self).__init__()
        if dnn_list[-1]!=1 or width > n_features:
            raise ValueError('dnn_list[-1]!=1 or width > n_features')
        init_ker_size=10
        self.cnn=Conv1d(4, out_channels, kernel_size=init_ker_size, stride=4,
                         padding=int((init_ker_size-1)/2), groups=1, bias=False, dilation=1)
        self.rnn=GRU(input_size=10*width,hidden_size=hidden,num_layers=n_layers,batch_first=True,bidirectional=True)
        self.last=[Linear(int(n_features/4),dnn_list[0])]
        
        for i in range(len(dnn_list)-1):
            self.last.append(BatchNorm1d(out_channels))
            self.last.append(ReLU())
            
            #self.last.append(Dropout())
            self.last.append(Linear(dnn_list[i],dnn_list[i+1]))
         
        self.last=Sequential(*self.last) 
        self.final=Linear(10,1)
        self.width=int(width)
    def forward(self, x  ):
        #n,c,l to n,l c
        #x = x.transpose(1,2)
        #x = x[:,0:x.size()[1]-x.size()[1]%self.width,:].contiguous()
         
        #x = x.view(x.size()[0],int(x.size()[1]/self.width),x.size()[2]*self.width)
        x = self.cnn(x)
        #x = self.rnn(x)[0]
        #x = x.transpose(1,2)
         
        x = self.last(x)
         
        x = torch.squeeze(x,dim=-1)
        x = self.final(x)
         
        
        return x
    def calc_size(self):
        1
        
class cnn59(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self):
        #N input_channels,C channels,L length
        super(cnn59, self).__init__()
        outsize0=1120
        self.cnn1=Conv1d(in_channels=4,out_channels=128,kernel_size=6,stride=1,dilation=1,padding=0)
        self.pool1=AdaptiveMaxPool1d(350)
        self.cnn2=Conv1d(in_channels=128,out_channels=32,kernel_size=8,stride=1,dilation=1,padding=0)
        self.pool2=AdaptiveMaxPool1d(35)
        self.cnnpart=Sequential(self.cnn1,self.pool1,self.cnn2,self.pool2)
        self.dnnpart=Sequential(Linear(outsize0,64),ReLU(),Dropout(),Linear(64,2),ReLU(),Dropout(),Linear(2,1))
         
    def forward(self, x  ):
         
        x=x.contiguous()
        x = self.cnnpart(x)
        
        x = x.view(x.size()[0],-1).contiguous()
        #print(x.size())
        x = self.dnnpart(x)
        
        return x
    def shrink(self,n,m):
        x = n
        for i in range(m):
            '''
            if self.di:
                x = np.floor((x+2*int((self.init_ker_size-1)/2)-2**i*(self.init_ker_size-1)-1)/self.stride+1)
                
            else:
            '''
            x = int( (x+2*int((self.init_ker_size-1)/2)-self.init_ker_size)/self.stride[0]+1)
             
        return x        

def getgene_reverse( x,rev_idx1,rev_idx2):
        return x.index_select(2,rev_idx1).index_select(1,rev_idx2)    

       
def print_info(loader,mod,device,text,quiet=False,pair=False):
     
    predvalue=[]
    realvalue=[]
    for datax,datay in loader:
        #datax,datay = seqtrain.get_pack(k%seqtrain.length,int(k/seqtrain.length)) 
         
        datax = datax.to(device)
        datay = datay.to(device)
         
        output = mod(datax)
        if output.dim()>1:
            output=  output.squeeze(-1)
        if datay.dim()>1:
            datay=  datay.squeeze(-1)
        if pair==True:
            rev_idx=torch.tensor( [i for i in range(datax.size()[-1]-1 , -1, -1)] ).to(device)
            rev2=torch.tensor([1,0,3,2]).to(device)
            datax_rev=getgene_reverse(datax,rev_idx,rev2)
            
            output_rev = mod(datax_rev)
            if output_rev.dim()>1:
                output_rev=  output_rev.squeeze(-1)
            output=0.5*( output+output_rev)
        if not ( datay.size()==output.size() ):
            print('err')
            print('datay.size() {}'.format(datay.size()))
            print('output.size() {}'.format(output.size()))
        predvalue.append(output.cpu().data.numpy().reshape(-1))
        realvalue.append(datay.cpu().data.numpy())
         
         
    #torch.save({'pred':predvalue,'real':realvalue},'train.pt')
# =============================================================================
    cor = np.corrcoef(np.concatenate(predvalue),np.concatenate(realvalue))[0,1]
    #if math.isnan(cor):  
    #    print('nan detected!')
    #    print(predvalue)
    if quiet==False:
        print('{} R square = {}, corrcoef = {}'.format(text,cor**2, cor))
    return predvalue,realvalue 




class NewGenewiseloader(Dataset):
    """return both the training and test dataset."""

    def __init__(self,  tissue  ,chrom ,len_per,\
                 seq_folder='/media/yilun/Elements/RefSeQ/',exp_folder='/media/yilun/Elements/DeepSeQ/expr/avg/'\
,in_mem=False,rev_train=0,start=0,end=0 ):
        """
        Args:
            tissue:values:0,1,2,3, ('Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood')
            exp_type:values: 0,1,2, 
            np (float) : percentage of number of patients selected.
            chrom in [6,8]
            len_per (float) :percentage of dna sequences selected
            n_train_gene_rate(float): ratio of the genes that are in the training data set, the 
            #rest are in the test data set.
            #seq folder '/home/yilun/dna/seqdata/chr8'
            #read seq data a = pd.read_csv('patient_id.txt',sep='\t') a.iloc[i,1] the first column at row i,
            #a.iloc[i,2] the second column at row i,a.iloc[:,0] is the gene-name list.
            #exp_folder '/home/yilun/dna/exp_nor_rbe/','NORM_RM.txt',
            #'/home/yilun/dna/exp_unnor_rbe/' ,'UNNORM_RM.txt', '/home/yilun/dna/exp_raw/','RAW.txt'
            #read expdata b = pd.read_csv('name.txt',sep='\t') b.iloc[i,0] is the expression value for
            first patient at gene i,
             b.index is the gene-name list,which is the same as a.iloc[:,0].b.columns is the list
             patient_id. The patients in this list will be fetched for the corresponding dna sequence in
             seq folder /home/yilun/dna/seqdata/SAMPLE2GENOTYPE.DICT
             
            #traindataset=1output training dataset
            #=2,output val dataset
            #=3,output test dataset
             Note:
                 1.all pd.read_csv must have sep='\t'
                 
            rev_train: 0 plain 1 augmented 2 channel concatenation 3 shape concatenation
        """
        #first columne of dicttable, is column names of exp_table
        #second columne of dicttable, is h5.keys()
        #os.listdir() files for seq
         
        
             
        self.len_percent=len_per
        self.rev_train=rev_train
        self.start=start
        self.end=end
         
        if ( not (len_per<=1 and len_per>=0)):
            raise ValueError( 'argument tissue in [0,1,2,3] and \
                             exp_type in [0,1,2] and np<=1 and np>=0 does not hold!')
            
        self.TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood',\
                'Adipose_Subcutaneous','Brain_Hippocampus','Esophagus_Mucosa','Prostate','Adipose_Visceral_Omentum','Brain_Hypothalamus'\
                ,'Skin_Not_Sun_Exposed_Suprapubic','Esophagus_Muscularis','Adrenal_Gland','Brain_Nucleus_accumbens_basal_ganglia','Fallopian_Tube',\
                'Skin_Sun_Exposed_Lower_leg','Artery_Aorta','Brain_Putamen_basal_ganglia','Heart_Atrial_Appendage','Small_Intestine_Terminal_Ileum',\
                'Artery_Coronary','Brain_Spinal_cord_cervical_c-1','Heart_Left_Ventricle','Spleen','Artery_Tibial','Brain_Substantia_nigra',\
                'Kidney_Cortex','Stomach','Bladder','Liver','Testis',\
                'Brain_Amygdala','Cells_EBV-transformed_lymphocytes','Lung','Thyroid','Brain_Anterior_cingulate_cortex_BA24','Cells_Transformed_fibroblasts',\
                'Minor_Salivary_Gland','Uterus','Brain_Caudate_basal_ganglia','Cervix_Ectocervix','Muscle_Skeletal','Vagina',\
                'Brain_Cerebellar_Hemisphere','Cervix_Endocervix','Nerve_Tibial','Brain_Cerebellum','Colon_Sigmoid','Brain_Cortex','Pancreas',\
                'Colon_Transverse','Brain_Frontal_Cortex_BA9','Esophagus_Gastroesophageal_Junction','Pituitary']
        
        
         
        
         
        exp_folder_dir_list=[]
        for j in chrom:
            #EXP_FOLDER= exp_folder+self.TISSUE[tissue]+'/chr{}_RAW.csv'.format(j) in avg
            EXP_FOLDER= exp_folder+self.TISSUE[tissue]+'/chr{}.csv'.format(j) #in avg_0
            exp_folder_dir_list.append(EXP_FOLDER)
         
        
        if os.name!='nt':
            exp_table=pd.concat([ pd.read_csv(exp_folder_dir_list[j],index_col=0,sep='\,',engine='python') for j in range(len(chrom))])
        else:
            exp_table=pd.read_csv('C:/tmp/NORM_RM.txt',sep='\t') 
        #print(exp_table )
        
        
         
        
        self.train=[]
        self.h5 = []#a list with each of its elements storing the seq data from one chrom
        for i1 in chrom:
            self.h5.append( h5py.File(seq_folder+'ref_chr{}.h5'.format(i1 ), "r") )
         
        h5_keys=[]
        for j in range(len(self.h5)):
            h5_keys=h5_keys+list(self.h5[j].keys())
        #print('exp len {} , key len {}'.format(exp_table.shape[0],len(h5_keys)))
        exp_table=exp_table.take([i for i in range(exp_table.shape[0]) if exp_table.index[i] in h5_keys],axis=0)
        #filter out low-expressed genes
        #exp_table=exp_table[exp_table>exp_table.quantile(0.2)].dropna()
        
        #print(exp_table)
        self.chr_list=[]
        for k in range(len(chrom)):
            #chrom index  for each gene
            self.chr_list+=list(k*np.ones(sum([i in  exp_table.index for i in self.h5[k].keys()] ),dtype=int) )
         
        #print('num_genes :{}'.format(exp_table.shape[0]))
        self.exp_shape=[exp_table.shape[0],exp_table.shape[1]]
         
        self.total_num_genes = exp_table.shape[0]
        self.in_mem=in_mem
         
        #self.dfy =  torch.from_numpy( exp_table.values.reshape(-1,order='F').astype(np.float32) ).type(torch.float32)
         
        
        self.dfy =    torch.from_numpy( exp_table.values.astype(np.float32) ).type(torch.float32)  
        
             
        
        self.gene_keys= list(exp_table.index) 
        self.chr_list= self.chr_list 
        
        #print('choosen={},unchoosen{}'.format(choosen,unchoosen))
           
        
        
        if self.start==0 and self.end==0:
            print('start site :{},end site :{},len :{}'.format(50000-int(50000*self.len_percent),50000+int(50000*self.len_percent),1+2*int(50000*self.len_percent)))
            self.dna_len=2*int(50000*self.len_percent)+1
        else:
            print('start site :{},end site :{},len :{}'.format(50000-int(50000*self.start),50000+int(50000*self.end),1+int(50000*start)+int(50000*end) ))
            self.dna_len=1+int(50000*start)+int(50000*end)
        idx = torch.tensor( [i for i in range(int(self.dna_len) , -1, -1)] )
        self.idx = idx
        if in_mem==True:
            self.train=[]
            for idx in range(len(self.dfy)):
                #print('idx {}'.format(idx))
                chr_num = self.chr_list[idx]
                
                #debug code
                '''
                if idx<=1:
                    print(chr_num)
                    print(self.gene_keys[idx])
                    print(list(self.h5[chr_num].keys())[:10])
                    print(self.gene_keys[idx] in list(self.h5[chr_num].keys()) )
                    print(torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() )
                '''   
                #end debug code
                
                #print(torch.from_numpy(self.h5[chr_num*len(self.section)+sec_num][self.namedict[self.patient_list[patient_num]]][self.gene_keys[genenum]][:][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float())
                if start==0 and end==0:
                    t_a=torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() 
                     
                        
                #print(t_a.size())#C,L,c=4
                else:
                    t_a=torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*start):50000+int(50000*end)+1]).float() 
                     
                        
                        
                if rev_train==0:
                    self.train.append( t_a ) #output size :#genes,4,seq_len
                if rev_train==1 or rev_train==4:
                    self.train.append( t_a )
                    self.train.append( self.reverse(t_a))
                if rev_train==2:#double length
                    self.train.append( torch.cat((t_a,self.reverse(t_a)),dim=1) )
                if rev_train==3:#8 channels
                    self.train.append( torch.cat((t_a,self.reverse(t_a)),dim=0) ) 
             
            self.dfx=torch.stack(self.train)
         
 
         

    def __len__(self):
        if  self.rev_train ==0:
            return len(self.dfy)
        else:#1,4
            return 2*len(self.dfy)

    def __getitem__(self, idx):
        if self.in_mem:
            if not (self.rev_train==1 or self.rev_train==4):
                return self.dfx[idx],self.dfy[idx]
            else:
                 
                return self.dfx[idx],self.dfy[int(idx/2)]
        else:
            if self.rev_train==0 :
                 
                chr_num = self.chr_list[idx]
                if self.start==0 and self.end==0:
                    return torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() ,self.dfy[idx]
                else:
                    return torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*self.start):50000+int(50000*self.end)+1]).float() ,self.dfy[idx]
                
            else:
                if self.rev_train==1 or self.rev_train==4:
                    half_idx=int(idx/2)
                    chr_num = self.chr_list[half_idx]
                    if idx%2==0:#positive chain
                        return torch.from_numpy(self.h5[chr_num][self.gene_keys[half_idx]][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() ,self.dfy[half_idx]
                    else:#negative
                        return self.reverse(torch.from_numpy(self.h5[chr_num][self.gene_keys[half_idx]][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() ),self.dfy[half_idx]
                 
            
            
            
    def reverse(self,x):
        return x.index_select(1,self.idx).index_select(0,torch.tensor([1,0,3,2]))
    
    def reverse_0(self,x ):
        return x.index_select(2,torch.tensor(list(range(x.size()[-1]-1,-1,-1)))).index_select(1,torch.tensor([1,0,3,2]))
    
 
    

class NewGenewiseloader_chunk(Dataset):
    """return both the training and test dataset."""

    def __init__(self,  tissue  ,chrom ,len_per,\
                 seq_folder='/media/yilun/Elements/RefSeQ/',exp_folder='/media/yilun/Elements/DeepSeQ/expr/avg/'\
,in_mem=False,rev_train=0,chunk=1 ):
        """
        Args:
            tissue:values:0,1,2,3, ('Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood')
            exp_type:values: 0,1,2, 
            np (float) : percentage of number of patients selected.
            chrom in [6,8]
            len_per (float) :percentage of dna sequences selected
            n_train_gene_rate(float): ratio of the genes that are in the training data set, the 
            #rest are in the test data set.
            #seq folder '/home/yilun/dna/seqdata/chr8'
            #read seq data a = pd.read_csv('patient_id.txt',sep='\t') a.iloc[i,1] the first column at row i,
            #a.iloc[i,2] the second column at row i,a.iloc[:,0] is the gene-name list.
            #exp_folder '/home/yilun/dna/exp_nor_rbe/','NORM_RM.txt',
            #'/home/yilun/dna/exp_unnor_rbe/' ,'UNNORM_RM.txt', '/home/yilun/dna/exp_raw/','RAW.txt'
            #read expdata b = pd.read_csv('name.txt',sep='\t') b.iloc[i,0] is the expression value for
            first patient at gene i,
             b.index is the gene-name list,which is the same as a.iloc[:,0].b.columns is the list
             patient_id. The patients in this list will be fetched for the corresponding dna sequence in
             seq folder /home/yilun/dna/seqdata/SAMPLE2GENOTYPE.DICT
             
            #traindataset=1output training dataset
            #=2,output val dataset
            #=3,output test dataset
             Note:
                 1.all pd.read_csv must have sep='\t'
                 
            rev_train: 0 plain 1 augmented 2 channel concatenation 3 shape concatenation
        """
        #first columne of dicttable, is column names of exp_table
        #second columne of dicttable, is h5.keys()
        #os.listdir() files for seq
         
        
        self.chunk=chunk
        self.len_percent=len_per
        self.rev_train=rev_train
        
        if ( not (len_per<=1 and len_per>=0)):
            raise ValueError( 'argument tissue in [0,1,2,3] and \
                             exp_type in [0,1,2] and np<=1 and np>=0 does not hold!')
            
        self.TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood',\
                'Adipose_Subcutaneous','Brain_Hippocampus','Esophagus_Mucosa','Prostate','Adipose_Visceral_Omentum','Brain_Hypothalamus'\
                ,'Skin_Not_Sun_Exposed_Suprapubic','Esophagus_Muscularis','Adrenal_Gland','Brain_Nucleus_accumbens_basal_ganglia','Fallopian_Tube',\
                'Skin_Sun_Exposed_Lower_leg','Artery_Aorta','Brain_Putamen_basal_ganglia','Heart_Atrial_Appendage','Small_Intestine_Terminal_Ileum',\
                'Artery_Coronary','Brain_Spinal_cord_cervical_c-1','Heart_Left_Ventricle','Spleen','Artery_Tibial','Brain_Substantia_nigra',\
                'Kidney_Cortex','Stomach','Bladder','Liver','Testis',\
                'Brain_Amygdala','Cells_EBV-transformed_lymphocytes','Lung','Thyroid','Brain_Anterior_cingulate_cortex_BA24','Cells_Transformed_fibroblasts',\
                'Minor_Salivary_Gland','Uterus','Brain_Caudate_basal_ganglia','Cervix_Ectocervix','Muscle_Skeletal','Vagina',\
                'Brain_Cerebellar_Hemisphere','Cervix_Endocervix','Nerve_Tibial','Brain_Cerebellum','Colon_Sigmoid','Brain_Cortex','Pancreas',\
                'Colon_Transverse','Brain_Frontal_Cortex_BA9','Esophagus_Gastroesophageal_Junction','Pituitary']
        
        
         
        
         
        exp_folder_dir_list=[]
        for j in chrom:
            EXP_FOLDER= exp_folder+self.TISSUE[tissue]+'/chr{}.csv'.format(j)
            exp_folder_dir_list.append(EXP_FOLDER)
         
        
        if os.name!='nt':
            exp_table=pd.concat([ pd.read_csv(exp_folder_dir_list[j],index_col=0,sep='\,',engine='python') for j in range(len(chrom))])
        else:
            exp_table=pd.read_csv('C:/tmp/NORM_RM.txt',sep='\t') 
        #print(exp_table )
        
        self.train=[]
        self.h5 = []#a list with each of its elements storing the seq data from one chrom
        for i1 in chrom:
            self.h5.append( h5py.File(seq_folder+'ref_chr{}.h5'.format(i1 ), "r") )
         
        h5_keys=[]
        for j in range(len(self.h5)):
            h5_keys=h5_keys+list(self.h5[j].keys())
        #print('exp len {} , key len {}'.format(exp_table.shape[0],len(h5_keys)))
        exp_table=exp_table.take([i for i in range(exp_table.shape[0]) if exp_table.index[i] in h5_keys],axis=0)
        #print(exp_table)
        self.chr_list=[]
        for k in range(len(chrom)):
            #chrom index  for each gene
            self.chr_list+=list(k*np.ones(sum([i in  exp_table.index for i in self.h5[k].keys()] ),dtype=int) )
         
        #print('num_genes :{}'.format(exp_table.shape[0]))
        self.exp_shape=[exp_table.shape[0],exp_table.shape[1]]
         
        self.total_num_genes = exp_table.shape[0]
        self.in_mem=in_mem
         
        #self.dfy =  torch.from_numpy( exp_table.values.reshape(-1,order='F').astype(np.float32) ).type(torch.float32)
         
        
        self.dfy =    torch.from_numpy( exp_table.values.astype(np.float32) ).type(torch.float32)  
        
             
        
        self.gene_keys= list(exp_table.index) 
        self.chr_list= self.chr_list 
        
        #print('choosen={},unchoosen{}'.format(choosen,unchoosen))
           
        idx = torch.tensor( [i for i in range(int(100000*len_per) , -1, -1)] )
        self.idx = idx
        if in_mem==True:
            self.train=[]
            for idx in range(len(self.dfy)):
                #print('idx {}'.format(idx))
                chr_num = self.chr_list[idx]
                
                #debug code
                '''
                if idx<=1:
                    print(chr_num)
                    print(self.gene_keys[idx])
                    print(list(self.h5[chr_num].keys())[:10])
                    print(self.gene_keys[idx] in list(self.h5[chr_num].keys()) )
                    print(torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float() )
                '''   
                #end debug code
                
                #print(torch.from_numpy(self.h5[chr_num*len(self.section)+sec_num][self.namedict[self.patient_list[patient_num]]][self.gene_keys[genenum]][:][:,50000-int(50000*self.len_percent):50000+int(50000*self.len_percent)+1]).float())
                t_a=torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(50000*chunk*self.len_percent):50000+int(50000*chunk*self.len_percent)+1]).float() 
                #print(t_a.size())#C,L,c=4
                if rev_train==0:
                    self.train.append( t_a ) #output size :#genes,4,seq_len
                if rev_train==1:
                    self.train.append( t_a )
                    self.train.append( self.reverse(t_a))
                if rev_train==2:#double length
                    self.train.append( torch.cat((t_a,self.reverse(t_a)),dim=1) )
                if rev_train==3:#8 channels
                    self.train.append( torch.cat((t_a,self.reverse(t_a)),dim=0) ) 
             
            self.dfx=torch.stack(self.train)
         
 
         

    def __len__(self):
        if  self.rev_train==0:
            return len(self.dfy)
        else:
            return 2*len(self.dfy)

    def __getitem__(self, idx):
        if self.in_mem:
            if  self.rev_train==0:
                return self.dfx[idx],self.dfy[idx]
            else:
                 
                return self.dfx[idx],self.dfy[int(idx/2)]
        else:
            
            chr_num = self.chr_list[idx]
            
            return torch.from_numpy(self.h5[chr_num][self.gene_keys[idx]][:,50000-int(self.chunk*50000*self.len_percent):50000+int(self.chunk*50000*self.len_percent)+1]).float() ,self.dfy[idx]
        
    def reverse(self,x):
        return x.index_select(1,self.idx).index_select(0,torch.tensor([1,0,3,2]))

def generate_sampler(N_sample,train_percent,val_percent,random_seed):
    np.random.seed(random_seed)
    choosen=np.random.choice(np.arange(N_sample),int(N_sample*(train_percent+val_percent)  ),replace=False)
    trainchoosen=np.random.choice(choosen,int(N_sample*(train_percent)  ),replace=False)
    valchoosen=np.setdiff1d(choosen,trainchoosen)
    testchoosen=np.setdiff1d(np.arange(N_sample),choosen) 
    return trainchoosen,valchoosen,testchoosen


        
def writeAvgExp(chrom,tissue,exp_folder='/media/yilun/Elements/DeepSeQ/expr/',pro_type=0):
    TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood',\
                'Adipose_Subcutaneous','Brain_Hippocampus','Esophagus_Mucosa','Prostate','Adipose_Visceral_Omentum','Brain_Hypothalamus'\
                ,'Skin_Not_Sun_Exposed_Suprapubic','Esophagus_Muscularis','Adrenal_Gland','Brain_Nucleus_accumbens_basal_ganglia','Fallopian_Tube',\
                'Skin_Sun_Exposed_Lower_leg','Artery_Aorta','Brain_Putamen_basal_ganglia','Heart_Atrial_Appendage','Small_Intestine_Terminal_Ileum',\
                'Artery_Coronary','Brain_Spinal_cord_cervical_c-1','Heart_Left_Ventricle','Spleen','Artery_Tibial','Brain_Substantia_nigra',\
                'Kidney_Cortex','Stomach','Bladder','Liver','Testis',\
                'Brain_Amygdala','Cells_EBV-transformed_lymphocytes','Lung','Thyroid','Brain_Anterior_cingulate_cortex_BA24','Cells_Transformed_fibroblasts',\
                'Minor_Salivary_Gland','Uterus','Brain_Caudate_basal_ganglia','Cervix_Ectocervix','Muscle_Skeletal','Vagina',\
                'Brain_Cerebellar_Hemisphere','Cervix_Endocervix','Nerve_Tibial','Brain_Cerebellum','Colon_Sigmoid','Brain_Cortex','Pancreas',\
                'Colon_Transverse','Brain_Frontal_Cortex_BA9','Esophagus_Gastroesophageal_Junction','Pituitary']
     
     
    exp_folder_dir_list=[]
    for j in chrom:
        if pro_type in [0,100]:
            EXP_FOLDER=exp_folder+'chr{}/exp_raw/'.format(j)
            exp_folder_dir_list.append(EXP_FOLDER+TISSUE[tissue]+'/RAW.txt')
        if pro_type in [1,11]:
            EXP_FOLDER=exp_folder+'chr{}/exp_nor_rbe/'.format(j)
            exp_folder_dir_list.append(EXP_FOLDER+TISSUE[tissue]+'/NORM_RM.txt')
            

        if pro_type in [2,12]:
            EXP_FOLDER=exp_folder+'chr{}/exp_unnor_rbe/'.format(j)
            exp_folder_dir_list.append(EXP_FOLDER+TISSUE[tissue]+'/UNNORM_RM.txt')
            

        
        if not path.exists('{}avg_{}/{}'.format(exp_folder,pro_type,TISSUE[tissue])):
            os.system('mkdir {}avg_{}/{}'.format(exp_folder,pro_type,TISSUE[tissue]))
    for j in range(len(chrom)):
        exp_table=pd.read_csv(exp_folder_dir_list[j],sep='\t')
        if pro_type in [0,11,12]:
            table_val=10**(exp_table.values)-0.1
            val=np.log(table_val.mean(axis=1))/np.log(10)
        if pro_type in [1,2]:
            table_val=exp_table.values
            val=table_val.mean(axis=1)
        if pro_type in [100]:
            table_val=10**(exp_table.values)-0.1
            val= table_val.mean(axis=1) 
        name=exp_table.index
        pd.DataFrame(data=val,index=name).to_csv(exp_folder+'avg_{}/{}/chr{}.csv'.format(pro_type,TISSUE[tissue],chrom[j]))
             
        

                 
class cnn_block2(torch.nn.Module):
    
    #L :seq_len
    # in_channels : 4
    # out_channels: output channels
    #out_L: output shapes
    # stride: for cnn
    #dilation: for cnn
    #pool_size: adpative pooling size
    #
    def __init__(self,L,in_channels,out_channels,ker_size,stride,dilation,pool_size,filter_type ):
        #N input_channels,C channels,L length
        super(cnn_block2, self).__init__()
        if filter_type=='cnn':
            self.layer=Conv1d(in_channels, out_channels, kernel_size=ker_size, stride=stride,
                                 padding=int((ker_size-1)/2), groups=1, bias=True, dilation=dilation)
        if filter_type=='pool':
            self.layer=MaxPool1d(in_channels, out_channels, kernel_size=ker_size, stride=stride,
                                 padding=int((ker_size-1)/2), dilation=dilation)
        self.out_L= int ( (L+2*int((ker_size-1)/2)-(ker_size-1)*dilation-1) / stride ) +1  
    def forward(self, x  ):
        y = self.layer(x)
        
        return y
        
        
class mix_C_layer(torch.nn.Module):
    
    #L :seq_len
    # in_channels : 4
    # out_channels: output channels
    #out_L: output shapes
    # stride: for cnn
    #dilation: for cnn
    #pool_size: adpative pooling size
    #
    def __init__(self,L,in_channels,out_channels,ker_size,stride,dilation,pool_size  ):
        #N input_channels,C channels,L length
        super(mix_C_layer, self).__init__()
        self.layer_list=ModuleList()
        for s in stride:
            for k in ker_size:
                for d in dilation:
                    pad_len=(k-1)*d+1
                    self.layer_list.append(\
                    Sequential( Conv1d(in_channels, out_channels, kernel_size=k, stride=s,\
        padding=int(pad_len/2), groups=1, bias=True, dilation=d), ReLU()))
         
         
    def forward(self, x  ):
        y =torch.stack([self.layer_list[j](x) for j in len(self.layer_list)],dim=-1)
        return y    
    
class mix_L_layer(torch.nn.Module):
    
    #L :seq_len
    # in_channels : 4
    # out_channels: output channels
    #out_L: output shapes
    # stride: for cnn
    #dilation: for cnn
    #pool_size: adpative pooling size
    #
    def __init__(self,L,in_channels,out_channels,ker_size,stride,dilation,pool_size,filter_type ):
        #N input_channels,C channels,L length
        super(mix_L_layer, self).__init__()
        if filter_type=='cnn':
            self.layer=Conv1d(in_channels, out_channels, kernel_size=ker_size, stride=stride,
                                 padding=int((ker_size-1)/2), groups=1, bias=True, dilation=dilation)
        if filter_type=='pool':
            self.layer=MaxPool1d(in_channels, out_channels, kernel_size=ker_size, stride=stride,
                                 padding=int((ker_size-1)/2), dilation=dilation)
        #self.out_L= int ( (L+2*int((ker_size-1)/2)-(ker_size-1)*dilation-1) / stride ) +1  
    def forward(self, x  ):
        y = self.layer(x)
        
        return y    

class test_mix(torch.nn.Module):
    
    #L :seq_len
    # in_channels : 4
    # out_channels: output channels
    #out_L: output shapes
    # stride: for cnn
    #dilation: for cnn
    #pool_size: adpative pooling size
    #
    def __init__(self,L,in_channels,out_channels,ker_size,stride,dilation,pool_size,filter_type ):
        #N input_channels,C channels,L length
        super(test_mix, self).__init__()
        self.layer1=mix_C_layer(L,in_channels,out_channels,ker_size,stride,dilation,pool_size)
        self.act1=ReLU()
        self.linear1=Linear((len(ker_size)*len(dilation)*len(stride)),1)
        self.bn1=BatchNorm1d(out_channels)
        self.drop=Dropout()
        self.linear2=Linear(out_channels,1)
        #self.out_L= int ( (L+2*int((ker_size-1)/2)-(ker_size-1)*dilation-1) / stride ) +1  
    def forward(self, x  ):
        y =self.layer1(x)#N,C,L
        y=self.linear1(y)
        y=y.squeeze(-1)
        
        y=self.bn1(y)
        y = self.act1(y)
        y=self.drop(y)
        z = self.linear2(y)
        return z    
    

def getfeature_xg(mod,loader,device,mydataset,saveid):
    #input model,loader,to get the sample_size*20020 output features
    all_x=[]
    all_y=[] 
    batchid=0
    for datax,datay in loader :
        batchid =batchid+1
        print('batch {}'.format(batchid))
        save_x=[]#200,batch, 2002
        mid=int(datax.size()[-1] /2)
        
        all_y.append(datay.cpu().data.numpy())
        for j in range(200):
            
            binx=datax[:,:,mid-200*100-900+j*200:mid-200*100+900+(j+1)*200]
            binx_rv=mydataset.reverse_0(binx)   
            binx = binx.to(device)
            binx_rv = binx_rv.to(device)
             
            output=mod.forward(torch.unsqueeze(binx,2))
             
            output_rv=mod.forward(torch.unsqueeze(binx_rv,2))
            avg_output=0.5*( output_rv+output)#batchsize, 2002
            save_x.append(avg_output.cpu().data.numpy())
        mx=[]
        for ai in [0.01,0.02,0.05,0.1,0.2]:
            basepos=np.zeros(save_x[0].shape)
            baseneg=np.zeros(save_x[0].shape)
            for j in range(100):
                baseneg = baseneg + save_x[j]*np.exp(-ai*abs( j-99.5) )  
            mx.append(baseneg)
            for j in range(100,200):
                basepos = baseneg + save_x[j]*np.exp(-ai*abs(j-99.5))  
            mx.append(basepos)
        #mx: 10 * batch,2002
        np.save('xbatchid'+str(batchid)+'svid'+str(saveid),np.stack(mx))
        np.save('ybatchid'+str(batchid)+'svid'+str(saveid),np.stack(datay.cpu().data.numpy()))
    #train_x=np.swapaxes(np.concatenate(all_x,axis=1).astype(np.float32),0,1)#sample size,10,2002
    #train_x=train_x.reshape(train_x.shape[0],-1)##sample size,20020
    #train_y=np.vstack(all_y).astype(np.float32)##sample size,1
    #return train_x,train_y

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))
    
class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    
    
class basenji(nn.Module):
    def __init__(self):
        super(basenji, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    
    
    
    
    
    
    
    
    


class downsample0(torch.nn.Module):
    
    #downsample changes the channels and width size
    def __init__(self,in_channels,out_channels,ker,stride,dilation,ispool,drop):
        #N input_channels,C channels,L length
        super(downsample0, self).__init__()
        if ispool==0:
            self.F= Sequential(BatchNorm1d(in_channels),Conv1d(\
    in_channels=in_channels,out_channels=out_channels,kernel_size=ker,\
    stride=stride,dilation=dilation,padding=0,bias=False),\
                    ReLU())
        else:
            self.F=  MaxPool1d(\
    kernel_size=ker,stride=stride,dilation=dilation,padding=0) 
    def forward(self, x  ):
       
        return self.F(x)

class resconv(torch.nn.Module):
    
     #convolutional residual block,with no extra input
    def __init__(self,in_channels,depth,ker,stride,dilation,drop):
        #N input_channels,C channels,L length
        super(resconv, self).__init__()
        assert ker%2==1 or dilation %2==0  
        self.depth=depth
        self.F=ModuleList()
        if drop==True:
            for j in range(depth):
                self.F.append( Sequential( BatchNorm1d(in_channels),Conv1d(\
    in_channels=in_channels,out_channels=in_channels,kernel_size=ker,stride=stride,\
    dilation=dilation,padding=int(dilation*(ker-1)/2),bias=False),\
                    ReLU(),Dropout())) 
        else:
            for j in range(depth):
                self.F.append( Sequential( BatchNorm1d(in_channels),Conv1d(\
    in_channels=in_channels,out_channels=in_channels,kernel_size=ker,stride=stride,\
    dilation=dilation,padding=int(dilation*(ker-1)/2),bias=False),\
                    ReLU())) 
         
    def forward(self, x ):
        x0 = x
        for j in range(self.depth):
            x = self.F[j](x)
        return x0 + x

class resNextbranch(torch.nn.Module):
    
     #bottleneck,multi-branch
    def __init__(self,in_channels,inter_channels,depth,ker,stride,dilation,drop):
        #N input_channels,C channels,L length
        super(resNextbranch, self).__init__()
        assert ker%2==1 or dilation %2==0  
        self.depth=depth
        self.F=ModuleList()
        self.F.append( Sequential( bncvblock(in_channels=in_channels,\
  out_channels=inter_channels,ker_size=ker[0],drop=drop ),bncvblock(in_channels=in_channels,\
  out_channels=inter_channels,ker_size=ker[1],drop=drop ),bncvblock(in_channels=in_channels,\
  out_channels=inter_channels,ker_size=ker[2],drop=drop ))) 
         
         
    def forward(self, x ):
         
        return self.F(x)

class bncvblock(torch.nn.Module):
    
     #bottleneck,multi-branch
      #only channel variant,changes from in_channels to out_channels
    def __init__(self,in_channels,out_channels,kernel_size,drop,preact):
        #N input_channels,C channels,L length
        super(bncvblock, self).__init__()
        
         
        self.F=ModuleList()
        if preact==False:
            if drop==True:
                 self.F= Sequential( BatchNorm1d(in_channels),Conv1d(\
        in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,\
        dilation=1,padding=int((kernel_size-1)/2),bias=False),\
                        ReLU(),Dropout()) 
            else:
                 self.F= Sequential( BatchNorm1d(in_channels),Conv1d(\
        in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,\
        dilation=1,padding=int((kernel_size-1)/2),bias=False),\
                        ReLU()) 
        else:
            if drop==True:
                 self.F= Sequential( BatchNorm1d(in_channels),ReLU(),Conv1d(\
        in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,\
        dilation=1,padding=int((kernel_size-1)/2),bias=False),\
                        Dropout()) 
            else:
                 self.F= Sequential( BatchNorm1d(in_channels),ReLU(),Conv1d(\
        in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=1,\
        dilation=1,padding=int((kernel_size-1)/2),bias=False)) 
         
    def forward(self, x ):
         
        return self.F(x)

class resNextblock(torch.nn.Module):
    
     #bottleneck,multi-branch
      #size invariant
    def __init__(self,in_channels,inter_channels,out_channels,C,ker_size_list,drop,preact):
        #N input_channels,C channels,L length
        super(resNextblock, self).__init__()
         
         
        self.F=ModuleList()
        for i in range(C):
            self.F.append( Sequential( bncvblock(in_channels=in_channels,\
      out_channels=inter_channels,ker_size=ker_size_list[0],drop=drop,preact=preact ),\
        bncvblock(in_channels=inter_channels,\
      out_channels=inter_channels,ker_size=ker_size_list[1],drop=drop,preact=preact ),\
        bncvblock(in_channels=inter_channels,\
      out_channels=out_channels,ker_size=ker_size_list[2],drop=drop,preact=preact ))) 
         
         
    def forward(self, x ):
        y=self.F[0](x)
        for f in self.F[1:]:
            y = y + self.f(x)
        return y

class resNextmodule(torch.nn.Module):
    
     #size invariant
     # block +depth
    def __init__(self,in_channels,inter_channels,out_channels,depth,C,ker_size_list,drop,connect,growth_channels=None):
        #N input_channels,C channels,L length
        super(resNextmodule, self).__init__()
         
        self.connect=connect 
        self.depth=depth
        self.F=ModuleList()
        if connect=='add':
            for i in range(depth):
                self.F.append( resNextblock(in_channels,inter_channels,out_channels,C,ker_size_list,drop)) 
        if connect=='cat':
            for i in range(depth):
                self.F.append( resNextblock(in_channels+i*growth_channels,inter_channels,in_channels,\
                                            C,ker_size_list,drop))        
         
         
    def forward(self, x ):
        if self.connect=="add":
            for i in range(self.depth):
                x  = x + self.F(x)
        if self.connect=="cat":
            for i in range(self.depth):
                x  = torch.cat( (x,self.F(x)),dim=1 )
        return x


class module0(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_len,in_channels,out_channels,ker,stride,depth ):
        #N input_channels,C channels,L length
        super(module0, self).__init__()
         
        self.cnn=Sequential(downsample0(in_channels=in_channels,out_channels=out_channels,\
                                       ker=ker[0],stride=stride[0],dilation=1,ispool=0,drop=False),\
    resconv(in_channels=out_channels,depth=depth,ker=ker[1],stride=stride[1],dilation=1,drop=False),\
    downsample0(in_channels=out_channels,out_channels=out_channels,\
               ker=ker[2],stride=stride[2],dilation=1,ispool=1,drop=False))
        
        self.compute_shape=lambda x: int( ( int( (x-ker[0])/stride[0] ) + 1 -ker[2] )/stride[2]) + 1
    def forward(self, x  ):
        x=self.cnn(x)
         
        return x

class mymodule(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_len,channels,ker,stride,dilation,depth,connect ):
        #N input_channels,C channels,L length
        super(mymodule, self).__init__()
         
        self.cnn=Sequential(downsample0(in_channels=channels[0],out_channels=channels[1],\
                                       ker=ker[0],stride=stride[0],dilation=1,ispool=0,drop=False))
    
    for i in range(len(depth)):
        
        self.cnn.append(resNextmodule(in_channels=channels[i,0],inter_channels=channels[i,1],\
   out_channels=channels[i,2],depth=depth[i],C=C[i],ker=ker[i+1],\
 drop=False,connect=connect))
        self.cnn.append(
    downsample0(in_channels=channels[i,2],out_channels=channels[i,3],\
               ker=ker[2],stride=stride[1],dilation=dilation[0],ispool=1,drop=False))
        
        self.compute_shape=lambda x: int( ( int( (x-ker[0])/stride[0] ) + 1 -ker[2] )/stride[2]) + 1
    def forward(self, x  ):
        x=self.cnn(x)
         
        return x




class standard(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_len,channels,ker,stride,depth,pool,dilation,li):
        #N input_channels,C channels,L length
        super(standard, self).__init__()
         
        self.layer0=module0(in_len=in_len,in_channels=2,\
        out_channels=channels[0] ,ker=[5,3,5],stride=[2,1,3] ,depth=2)
        L0=self.layer0.compute_shape(in_len)
        
         
        
        self.layer1=module0(in_len=L0,in_channels=channels[0],\
        out_channels=channels[1] ,ker=[5,3,5],stride=[2,1,3] ,depth=2)
        L1=self.layer1.compute_shape(L0)
        
         
        
        self.layer2=module0(in_len=L1,in_channels=channels[1],\
        out_channels=channels[2] ,ker=[5,3,5],stride=[2,1,3] ,depth=2)
        L2=self.layer2.compute_shape(L1)
        
         
        
        self.cnn=Sequential(self.layer0,self.layer1,self.layer2)
        
        self.li=Sequential(  Linear(channels[2]*L2,li[0]),BatchNorm1d(li[0]) ,ReLU(), Dropout(),\
                           Linear(li[0],li[1]),BatchNorm1d(li[1]) ,ReLU(),\
        Dropout(),Linear(li[1],1))
         
        
        
         
        
        
        
    def forward(self, x  ):
        x=self.cnn(x)
        x= x.view(x.size()[0],-1)
        x=self.li(x)
         
        return x

     
