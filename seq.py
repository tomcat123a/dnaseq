# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:25:58 2019

@author: Administrator
"""


import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,AdaptiveAvgPool1d,Linear,MSELoss,LSTM,GRU
from torch.nn import  LeakyReLU,ReLU
import torch.nn as nn

import pandas as pd
import time
import numpy as np
from drnn import DRNN 

from torch.utils.data import Dataset
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


def conv1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
class block0(torch.nn.Module):  #resnet
    expand=2
    def __init__(self,in_channels,out_channels,dilation,stride=1,first=False,res_on=True):
        #the first block will increase the channels even stride==1, or do downsampling and increasing channels 
        #even stride==1
        #if it is not the first block, then input x channels == out_channels, and then it will be shrunk to in_channels
        
        super(block0, self).__init__()
        
        if first==False:
            self.bn1=BatchNorm1d(out_channels)
            self.cv1=conv1(out_channels,in_channels)
        else:
            self.bn1=BatchNorm1d(in_channels)
            self.cv1=conv1(in_channels,in_channels)
        self.ac1= ReLU()
        
        self.bn2=BatchNorm1d(in_channels)
        self.ac2= ReLU()
        self.cv2=conv3(in_channels,in_channels,dilation=dilation,stride=stride)
        self.bn3=BatchNorm1d(in_channels)
        self.ac3= ReLU()
        self.cv3=conv1(in_channels,out_channels)
        if first==True or stride!=1:
            self.downsample=downsample(in_channels,out_channels,stride)
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
        
        x = self.bn1(x)
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
     
    def __init__(self,in_channels,out_channels,stride):
        #N input_channels,C channels,L length
        super(downsample, self).__init__()
        
        #self.bn1=BatchNorm1d(in_channels)
         
        self.cv1=conv1(in_channels,out_channels,stride=stride)
         
         
    def forward(self, x ):
        
        return self.cv1(x)


    
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
        #if int(in_channels/splits)!=in_channels/splits:
        #    raise ValueError('in_channels/splits is not an integer!')
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
                                          factor**(i+1)*in_channels),block_type=block_type,stride=2,
                                          depth=depth,dilation=2**i,**kwargs))
        if degrid == True:
            self.layer_list.append(layer0(in_channels=int(factor**n_layers*in_channels),out_channels=int(factor\
                                          **n_layers*in_channels),block_type=block_type,stride=1,
                                              depth=depth,dilation=1,res_on=False,**kwargs))
            
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
    def __init__(self, n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False):
        super(biDRNN, self).__init__()
        self.drnn1=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type=cell_type, batch_first=False)
        self.drnn2=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type=cell_type, batch_first=False)
        
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
    def __init__(self,in_channels,n_layers,n_features,depth,block_type,degrid,testrnn,cell_type,hidden_size,n_layers_rnn):
        #N input_channels,C channels,L length
        super(testcnn, self).__init__()
        factor=1.41
        self.rnn=testrnn
        self.cell_type=cell_type
        cnn_output_channels=int(in_channels*factor**(n_layers))
        cnn_output_len=int(self.shrink(n_features,n_layers))
        if cell_type=='LSTM':
            self.rnn=LSTM(input_size=cnn_output_channels,hidden_size=hidden_size,num_layers=n_layers_rnn,bidirectional=True,batch_first=True)
        if cell_type=='GRU':
            self.rnn = GRU(input_size=cnn_output_channels,hidden_size=hidden_size,num_layers=n_layers_rnn,bidirectional=True,batch_first=True)
        if cell_type=='biDRNN':
            self.rnn = biDRNN(n_input=cnn_output_channels,n_hidden=hidden_size,n_layers=n_layers_rnn ,cell_type='GRU', batch_first=True)
        self.prelayer=res(in_channels=in_channels,n_layers=n_layers,n_features=n_features,init_ker_size=7,\
              block_type=block_type,depth=depth,zero_init=True,degrid=degrid,tail=False)
        self.avdpool_cnn=AdaptiveAvgPool1d(1)
        self.fc_cnn = Linear(cnn_output_channels, 1)
        self.timedistributed_rnn=Linear(cnn_output_len,1)
        self.fc_rnn = Linear( 2*hidden_size, 1)
    def forward(self, x , rnn=True ):
        x = self.prelayer(x)
         
        x1 = self.avdpool_cnn(x)
        x1 = x1.squeeze(dim=-1)
        x1 = self.fc_cnn(x1)
        
        if self.rnn==True:
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
    """Face Landmarks dataset."""

    def __init__(self, csv_file_x, csv_file_y,gen=False,n=1000 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gen=gen
        if gen==False:
            
            self.dfx = pd.read_csv(csv_file_x)
            self.dfy = pd.read_csv(csv_file_y)
        else:
            n_seq = 40001
            n_channels = 4
            self.dfx=[]
            self.dfy=[]
            for i in range(n):
                x = np.zeros((  n_channels,n_seq ) )
                J = np.random.choice(n_channels, n_seq)
                x[J , np.arange(n_seq)] = 1
                self.dfx.append(x)
                self.dfy.append(sum(x[0:8,7])+0.5*sum(x[100:108,1])+\
                                2*sum(x[400:430,2])**2+0.05*sum(x[19900:200100,2:4])**2+0.1*np.random.rand(1))

# assign with advanced indexing
 
         

    def __len__(self):
        return len(self.dfx)

    def __getitem__(self, idx):
        if self.gen==False:
            return self.dfx.iloc[idx,],self.dfy.iloc[idx,]
        else:
            return torch.from_numpy(self.dfx[idx]).type(torch.float32),torch.from_numpy(self.dfy[idx]).type(torch.float32)
     
