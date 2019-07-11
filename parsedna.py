# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:55:33 2019

@author: Administrator
"""

import pandas as pd

import torch

#type of [1,0,0,0] or [0,0,0,0] must be float
mydic={"A" : [1., 0, 0, 0],
  "T" : [0., 1, 0, 0],
  "C" : [.0, 0, 1, 0],
  "G" : [0., 0, 0, 1],
  "-" : [0., 0, 0, 0],
  "." : [0., 0, 0, 0],
  "N" : [0.25, 0.25, 0.25, 0.25],
  "R" : [0.50, 0, 0, 0.50],
  "Y" : [0, 0.50, 0.50, 0],
  "S" : [0, 0, 0.50, 0.50],
  "W" : [0.50, 0.50, 0, 0],
  "K" : [0, 0.50, 0, 0.50],
  "M" : [0.50, 0, 0.50, 0],
  "B" : [0, 0.33, 0.33, 0.33],
  "D" : [0.33, 0.33, 0, 0.33],
  "H" : [0.33, 0.33, 0.33, 0],
  "V" : [0.33, 0, 0.33, 0.33]  }

 
    
def totensor(x,setidx,percent=1):
    #input size num_genes,2, and each element is a character seq with its len==seq_len ,output size:len(setidx),4,seq_len
    #return (N)
    mid=int((len(x.iloc[0,0])-1)/2)
    start=mid-int(mid*percent)
    end=mid+int(mid*percent)+1
    return torch.stack( [torch.transpose(\
    0.5*( torch.tensor( [mydic[i] for i in x.iloc[j,0][start:end]]) +\
         torch.tensor( [mydic[i] for i in x.iloc[j,1][start:end]] ) ) ,0,1) for j in setidx] ,dim=0 )
'''
p2=pd.DataFrame([['ATCG-.HHHHVYYYY','TACDWKBBBBMYYYY'],['ATCG-.BBBBMYYYY','TACDWKKKKKDYYYY']])    
p3=pd.concat((p2,p2,p2),0) 
per=0.2
per=0.4
per=0.6
per=0.8
'''   
totensor(p3,[0,1,2],per).size()==torch.Size([3, 4, 1+2*int(per*(len(p3.iloc[0,0])-1)/2)])

    #input two columns of strings dtype.pandas df
    
    
 
