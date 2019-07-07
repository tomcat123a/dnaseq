# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:02:05 2019

@author: Administrator
"""

import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,AdaptiveAvgPool1d,Linear,MSELoss,LSTM,GRU
from torch.nn import  LeakyReLU,ReLU
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import time
import numpy as np
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
    

def shrink(n,m):
    x = n
    for i in range(m):
        if floor(x/2)==x/2:
            x = x/2
        else:
            x = floor(x/2)+1 
    return int(x)
shrink(100,3)
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
    
#test file size test
inputa=torch.rand(3,12,10)
 
l1=branch0(in_channels=12,out_channels=24,stride=1,dilation=1,splits=1)
assert l1(inputa).size()==torch.Size([3,24,10])

l1=branch0(in_channels=12,out_channels=24,stride=1,dilation=1,splits=3)
assert l1(inputa).size()==torch.Size([3,24,10])

l1=branch0(in_channels=12,out_channels=24,stride=2,dilation=1,splits=2)
assert l1(inputa).size()==torch.Size([3,24,5])

lb0=block0(in_channels=12,out_channels=24,stride=1,dilation=1,first=True)
assert lb0(inputa).size()==torch.Size([3,24,10])

lb0=block0(in_channels=12,out_channels=24,stride=1,dilation=1,first=False)
inputb=torch.rand(3,24,10)
assert lb0(inputb).size()==inputb.size()

lb0=block0(in_channels=12,out_channels=24,stride=2,dilation=1,first=True)
assert lb0(inputa).size()==torch.Size([3,24,5])

lb0=block0(in_channels=12,out_channels=24,stride=1,dilation=2,first=False)
assert lb0(inputb).size()==inputb.size()

#res_on
lb0=block0(in_channels=12,out_channels=24,stride=1,dilation=1,first=True,res_on=False)
assert lb0(inputa).size()==torch.Size([3,24,10])

lb0=block0(in_channels=12,out_channels=24,stride=1,dilation=1,first=False,res_on=False)
inputb=torch.rand(3,24,10)
assert lb0(inputb).size()==inputb.size()

lb0=block0(in_channels=12,out_channels=24,stride=2,dilation=1,first=True,res_on=False)
assert lb0(inputa).size()==torch.Size([3,24,5])

 


lb0=block1(in_channels=12,out_channels=24,stride=1,dilation=1,first=True,splits=2)
assert lb0(inputa).size()==torch.Size([3,24,10])

lb0=block1(in_channels=12,out_channels=24,stride=1,dilation=1,first=False,splits=1)
inputb=torch.rand(3,24,10)
 
assert lb0(inputb).size()==inputb.size()

lb0=block1(in_channels=12,out_channels=24,stride=2,dilation=1,first=True,splits=2)
assert lb0(inputa).size()==torch.Size([3,24,5])

lb0=block1(in_channels=12,out_channels=24,stride=1,dilation=2,first=False,splits=4)
assert lb0(inputb).size()==inputb.size()

#res_on
lb0=block1(in_channels=12,out_channels=24,stride=2,dilation=1,first=True,splits=2,res_on=False)
assert lb0(inputa).size()==torch.Size([3,24,5])

lb0=block1(in_channels=12,out_channels=24,stride=1,dilation=2,first=False,splits=4,res_on=False)
assert lb0(inputb).size()==inputb.size()

assert inputa.size()==torch.Size([3,12,10])
lay0=layer0(in_channels=12,out_channels=24,block_type=block0,depth=3,dilation=1,stride=1)

assert lay0(inputa).size()==torch.Size([3,24,10])
lay0=layer0(in_channels=12,out_channels=24,block_type=block0,depth=3,dilation=1,stride=2)
assert lay0(inputa).size()==torch.Size([3,24,5])

lay0=layer0(in_channels=12,out_channels=24,block_type=block0,depth=3,dilation=[2,4,8],stride=2)
assert lay0(inputa).size()==torch.Size([3,24,5])

lay0=layer0(in_channels=12,out_channels=24,block_type=block0,depth=3,dilation=[2,4,8],stride=2,res_on=False)
assert lay0(inputa).size()==torch.Size([3,24,5])

lay0=layer0(in_channels=12,out_channels=24,block_type=block1,depth=3,dilation=1,stride=2,splits=2)
assert lay0(inputa).size()==torch.Size([3,24,5])

lay0=layer0(in_channels=12,out_channels=24,block_type=block1,depth=3,dilation=1,stride=1,splits=4)

assert lay0(inputa).size()==torch.Size([3,24,10])

lay0=layer0(in_channels=12,out_channels=24,block_type=block1,depth=3,dilation=1,stride=1,splits=4,res_on=False)
assert lay0(inputa).size()==torch.Size([3,24,10])

lay0=layer0(in_channels=12,out_channels=24,block_type=block1,depth=3,dilation=1,stride=2,splits=2)
assert lay0(inputa).size()==torch.Size([3,24,5])
lay0=layer0(in_channels=12,out_channels=24,block_type=block1,depth=3,dilation=[2,4,8],stride=2,splits=2)
assert lay0(inputa).size()==torch.Size([3,24,5]) 
#test res

inputx=torch.rand(5,4,100)
mod=res(in_channels=4,n_layers=3,n_features=100,init_ker_size=9,block_type=block0,depth=2,zero_init=False)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=0,n_features=100,init_ker_size=9,block_type=block0,depth=2,tail=False,zero_init=False)
assert mod(inputx).size()==inputx.size()
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block1,depth=3,zero_init=False,splits=1)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block1,depth=3,zero_init=False,splits=2)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=7,block_type=block0,depth=3,zero_init=True)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=7,block_type=block1,depth=2,zero_init=True,splits=1)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=5,block_type=block1,depth=2,zero_init=True,splits=2)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=1,n_features=4,init_ker_size=3,block_type=block0,depth=1,zero_init=False)
assert mod(inputx).size()==torch.Size([5,1])

mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=5,block_type=block1,depth=2,zero_init=True,degrid=False,splits=2)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False,degrid=False)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False,degrid=False)
assert mod(inputx).size()==torch.Size([5,1])
mod=res(in_channels=4,n_layers=1,n_features=4,init_ker_size=3,block_type=block0,depth=1,zero_init=False,degrid=False)
assert mod(inputx).size()==torch.Size([5,1])

mod=res(in_channels=4,n_layers=3,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False,degrid=False,tail=False)
assert mod(inputx).size()==torch.Size([5,11,13])
mod=res(in_channels=4,n_layers=0,n_features=4,init_ker_size=3,block_type=block0,depth=1,zero_init=False,degrid=False,tail=False)
assert mod(inputx).size()==torch.Size([5,4,100])
def calc_num_par(x):
    #calculate the total number of parameters for a pytorch nn.Module
    pytorch_total_params = sum(p.numel() for p in x.parameters() if p.requires_grad)
    return  pytorch_total_params
#data
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
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
seq=SeqDataset(None,None,True,1000)
 
seqtest=SeqDataset(None,None,True,200)
params = {'batch_size': 32,
          'shuffle': True,
          'pin_memory':True} 


train_loader=DataLoader(seq,**params) 
test_loader=DataLoader(seqtest,batch_size=30,shuffle=False,pin_memory=True)
model=res(in_channels=4,n_layers=9,n_features=40001,init_ker_size=7,\
          block_type=block0,depth=2,zero_init=True,degrid=False).to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001,amsgrad=True) 


train_loss_list=[]
test_loss_list=[]
count=0
maxit=100
itertime=0
for count in range(maxit):
    if count==0:
        print('batch size{}'.format(params['batch_size']))
        print('num of pars in model={}'.format(calc_num_par(model)))
    for data in train_loader:
        data[0] = data[0].to(device)
        data[1] = data[1].to(device) 
        #xdata=data
        #t_x=data.x.type(torch.float)
        #t_edge=data.edge_index
        #t_batch=data.batch
        #break
        
        t0=time.time()
        optimizer.zero_grad()
        
        output=model(data[0])
            #c_idx = c_index(output,torch.from_numpy(y_train),torch.from_numpy( censor_train))
        loss = MSELoss(reduction='sum')(output,data[1])#event=0,censored
        #train_loss_list.append(loss.cpu().data.numpy())
        t1=time.time()-t0
        #print('forward takes {}s'.format(t1))
        train_loss_list.append(loss.cpu().data.numpy())
        if itertime%100==0:
            print('train loss = {} at iter {}'.format(train_loss_list[-1],itertime))
        t0=time.time()
        loss.backward()
        t1=time.time()-t0
        
        optimizer.step() 
        itertime+=1
        #print('backward takes {}s'.format(t1))
     

total_test_loss=0
total_r2=0
total_cor=0
for data in test_loader:
    data[0] = data[0].to(device)
    data[1] = data[1].to(device) 
    output = model(data[0])
    total_cor = np.corrcoef(output.cpu().data.numpy().reshape(-1),data[1].cpu().data.numpy().reshape(-1))[0,1]
    loss = MSELoss(reduction='sum')(output,data[1]) 
    total_test_loss+=loss.cpu().data.numpy()
    total_r2+=data[1].var()*(data[1].size()[0]-1)
 
ratio = total_test_loss/total_r2
    
print('final R square = {}, corrcoef = {}'.format(ratio,total_cor))
