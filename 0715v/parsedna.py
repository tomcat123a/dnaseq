# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:55:33 2019

@author: Administrator
"""

import pandas as pd
import re
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

def chartoidx(x,str0):
     
    for name,vec in mydic.items():
        #print(name)
        idx=[i.start() for i in re.finditer(name,str0)]
        
        x[:,idx]=torch.unsqueeze(torch.tensor(mydic[name]),1).expand(4,len(idx))
    return x
    
def totensor(x,percent=1):
    #input size num_genes,2, and each element is a character seq with its len==seq_len ,output size:len(setidx),4,seq_len
    #return (N)
    #output size genes,4,seq_len
    x = x.values
    mid=int((len(x[0,0])-1)/2)
    start=mid-int(mid*percent)
    end=mid+int(mid*percent)+1
    num_genes=x.shape[0]
    '''
    return torch.stack( [torch.transpose(\
    0.5*( torch.tensor( [mydic[i] for i in x[j,0][start:end]]) +\
         torch.tensor( [mydic[i] for i in x[j,1][start:end]] ) ) ,0,1) for j in range(x.shape[0])] ,dim=0 )
    '''
    output = torch.zeros(num_genes,4,int(percent*100001))
    for i in range(num_genes):
        output[i]= 0.5*( chartoidx(output[i],x[i,0][start:end])  +  chartoidx(output[i],x[i,1][start:end]) )
    return output
    
'''
p2=pd.DataFrame([['ATCG-.HHHHVYYYY','TACDWKBBBBMYYYY'],['ATCG-.BBBBMYYYY','TACDWKKKKKDYYYY']])    
p3=pd.concat((p2,p2,p2),0) 
per=0.2
per=0.4
per=0.6
per=0.8
   
assert totensor(p3,per).size()==torch.Size([p3.shape[0], 4, 1+2*int(per*(len(p3.iloc[0,0])-1)/2)])

 
    #input two columns of strings dtype.pandas df
ao=torch.zeros(4,10)
ao.expand(torch.zeros(4,20).size())
torch.unsqueeze(ao,2).size()
chartoidx(ao,'ATCGB.A.A.').transpose(0,1)
    
 '''
