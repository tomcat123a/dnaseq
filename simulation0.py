# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:02:05 2019

@author: Administrator
"""

import torch
from torch.nn import Conv1d,BatchNorm1d, ModuleList ,Sequential,Identity
from torch.nn import  LeakyReLU,ReLU
  
#generate sequence data
#generate expression value
#build models
#bayes optimization
#architecture search
torch.randint((10,4))
class CONVMOD(torch.nn.Module):  
    def __init__(self ):
        super(CONVMOD, self).__init__()
        
         
        
        
    def forward(self, x ):
        
         
        return x
#implement resnet,densenet

def conv3(in_channels, out_channels, stride=1 , dilation=1):
    """3x1 convolution with padding"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=1, bias=False, dilation=dilation)


def conv1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
class block0(torch.nn.Module):  #resnet
    expand=2
    def __init__(self,in_channels,out_channels,di,stride=1,first=False):
        
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
        self.cv2=conv3(in_channels,in_channels,dilation=di,stride=stride)
        self.bn3=BatchNorm1d(in_channels)
        self.ac3= ReLU()
        self.cv3=conv1(in_channels,out_channels)
        if first==True or stride!=1:
            self.downsample=downsample(in_channels,out_channels,stride)
        else:
            self.downsample=None
    def forward(self, x ):
        #x.size() #N input_channels,C channels,L length
          #(N,C,L),C=2*in_channels
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
    
    def __init__(self,in_channels,out_channels,block_type,depth,di,stride,**kwargs):
        #N input_channels,C channels,L length
        super(layer0, self).__init__()
        if not isinstance(di,list):
            di=[di]*depth
        self.block_list=[]
        for i in range(depth):
            if i==0:
                first=True
                b_stride=stride
            else:
                first=False
                b_stride=1
            self.block_list+=[block_type(in_channels,out_channels,di[i],b_stride,first,**kwargs)]
        self.downsample=downsample(in_channels,stride=stride)
        self.seq=Sequential(*self.block_list)
    def forward(self, x ):
        x = self.downsample(x)
        x = self.seq(x) 
        #x = self.lin2(x)
        return x
    

    
class layer1(torch.nn.Module):  #resnet
    
    def __init__(self,in_channels,block_type,depth,di,**kwargs):
        #N input_channels,C channels,L length
        super(layer1, self).__init__()
        if not isinstance(di,list):
            di=[di]*depth
        self.block_list=[]
        for i in range(depth):
             block_list+=[block_type(in_channels,di[i],**kwargs)]
        self.downsample=downsample(in_channels,stride=2)
        #self.seq=Sequential(*block_list)
        self.H=[]
        self.convH=conv1(in_channels*2)
    def forward(self, x ):
        x = self.downsample(x)
        for i in range(depth):
            x = self.block_list(x)
            if i < depth-1:
                self.H.append(x)
        #x = self.lin2(x)
        return x
    
class block1(torch.nn.Module):  #densenet multibranch
    expand=2
    def __init__(self,in_channels,out_channels,di,stride,split,first=False):
        #N input_channels,C channels,L length
        super(block1, self).__init__()
        
        self.bn1=BatchNorm1d(out_channels)
        self.ac1= ReLU()
        if in_channels!=out_channels or first==True or stride!=1:
            self.downsample=downsample(in_channels,out_channels,stride)
        else:
            self.downsample=None
        self.branch_list=[]
        for i in range(split):
            self.branch_list = self.branch_list + [branch0(in_channels,out_channels,di,stride,split)]
         
    def forward(self, x ):
        if self.downsample is not None:
            x0 = self.downsample(x)
        else:
            x0 = x
        x = self.bn1(x)
        x = self.ac1(x)
        x = torch.cat( [self.branch_list[i](x) for i in range(split)],dim=1)
        x = x + x0
        #x = self.lin2(x)
        return x
    
class branch0(torch.nn.Module):  #densenet multibranch
    expand=2
    def __init__(self,in_channels,out_channels,di,stride,split):
        #N input_channels,C channels,L length
        super(branch0, self).__init__()
        if int(in_channels/split)!=in_channels/split:
            raise ValueError('in_channels/split is not an integer!')
        inter_channels=int(in_channels/split)
        self.cv1=conv1(in_channels,inter_channels)
        self.bn2=BatchNorm1d(inter_channels)
        self.ac2= ReLU()
        self.cv2=conv3(inter_channels,inter_channels,dilation=di,stride=stride)
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
    
12/2
inputa=torch.rand(3,12,10)
inputa=torch.rand(3,24,10)
l1=branch0(12,24,1,1,1)
l1(inputa).size()
l1=branch0(12,24,2,1,1)
l1(inputa).size()
l1=branch0(12,24,1,2,1)
l1(inputa).size()
l1=branch0(12,24,1,2,3)
l1(inputa).size()
Conv1d(12,2,kernel_size=1)
lb0=block0(12,24,1,1,True)
lb0(inputa).size()
inputb=torch.rand(3,24,10)
lb1=block0(12,24,1,1,False)
lb1(inputb).size()

