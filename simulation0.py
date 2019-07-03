# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:02:05 2019

@author: Administrator
"""

import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,AdaptiveAvgPool1d,Linear
from torch.nn import  LeakyReLU,ReLU
import torch.nn as nn
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
        if int(in_channels/splits)!=in_channels/splits:
            raise ValueError('in_channels/splits is not an integer!')
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
    
def get_size(x,n):
     
    for i in range(n):
        x =  floor((x-1)/2+1)
    return int(x )
class res(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_channels,n_layers,n_features,init_ker_size,block_type,depth,zero_init,**kwargs):
        #N input_channels,C channels,L length
        super(res, self).__init__()
        if init_ker_size%2!=1:
            raise ValueError('init_ker_size must be an odd number')
        self.layer_list=[]
        self.layer_list.append(Conv1d(in_channels, in_channels, kernel_size=init_ker_size, stride=1,
                     padding=int((init_ker_size-1)/2), groups=1, bias=False, dilation=1))
        for i in range(n_layers):
            self.layer_list.append(layer0(in_channels=2**i*in_channels,out_channels=2**(i+1)*in_channels,block_type=block_type,stride=2,
                                          depth=depth,dilation=2**i,**kwargs))
        self.layer_list.append(layer0(in_channels=2**n_layers*in_channels,out_channels=2**n_layers*in_channels,block_type=block_type,stride=1,
                                          depth=depth,dilation=2**i,res_on=False,**kwargs))
        self.seq=Sequential(*self.layer_list)
        self.avdpool=AdaptiveAvgPool1d(1)
        self.fc = Linear(2**n_layers*in_channels, 1)
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
    def forward(self, x ):
        print(x.size())
        x = self.seq(x)
        print(x.size())
        x = self.avdpool(x)
        print(x.size())
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
mod(inputx).size()==torch.Size([5,1])

mod=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block1,depth=3,zero_init=False,splits=1)
mod(inputx).size()==torch.Size([5,1])
out=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block1,depth=3,zero_init=False,splits=2)
mod(inputx).size()==torch.Size([5,1])
out=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=7,block_type=block0,depth=3,zero_init=True)
mod(inputx).size()==torch.Size([5,1])
out=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=7,block_type=block1,depth=2,zero_init=True,splits=1)
mod(inputx).size()==torch.Size([5,1])
out=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=5,block_type=block1,depth=2,zero_init=True,splits=2)
mod(inputx).size()==torch.Size([5,1])
out=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False)
mod(inputx).size()==torch.Size([5,1])
out=res(in_channels=4,n_layers=4,n_features=50,init_ker_size=9,block_type=block0,depth=3,zero_init=False)
mod(inputx).size()==torch.Size([5,1])



