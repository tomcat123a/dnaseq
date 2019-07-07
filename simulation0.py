# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:02:05 2019

@author: Administrator
"""

import torch
from seq import branch0,block0,block1,res,layer0,calc_num_par,SeqDataset,testrnn
from torch.utils.data import Dataset,DataLoader
from torch.nn import MSELoss
import time
import sys
import numpy as np
#data
   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

quiet=True
seq=SeqDataset(None,None,True,1000)
 
seqtest=SeqDataset(None,None,True,200)
params = {'batch_size': 16,
          'shuffle': True,
          'pin_memory':True} 


train_loader=DataLoader(seq,**params) 
test_loader=DataLoader(seqtest,batch_size=30,shuffle=False,pin_memory=True)
#model=res(in_channels=4,n_layers=9,n_features=40001,init_ker_size=7,\
#          block_type=block0,depth=2,zero_init=True,degrid=False).to(device)
cnn_para_list=\
[{'in_channels':4,'n_layers':3,'n_features':40001,'init_ker_size':7,'block_type':block0,\
 'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':3,'n_features':40001,'init_ker_size':7,'block_type':block0,\
 'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':4,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':4,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':5,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':5,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':6,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':6,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':7,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':7,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':8,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':8,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':9,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':9,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':3,'zero_init':True,'degrid':True,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':5,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':True},\
{'in_channels':4,'n_layers':7,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':True},\
{'in_channels':4,'n_layers':9,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':True,'tail':False,'testrnn':True},\
{'in_channels':4,'n_layers':5,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':False,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':7,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':False,'tail':False,'testrnn':False},\
{'in_channels':4,'n_layers':9,'n_features':40001,'init_ker_size':7,'block_type':block0,\
'depth':2,'zero_init':True,'degrid':False,'tail':False,'testrnn':False}]
#0-13 0,2,4,6,8,10,12 depth=2,1,3,5,7,9,11,13,depth=3,degrid=True,n_layers=3,4,5,6,7,8,9,testrnn=False
#14,15,16,n_layers=5,7,9,depth=2,testrnn=True
#17,18,19,n_layers=5,7,9,depth=2,testrnn=False,degrid=False
for ID in range(20):
    model=testrnn(in_channels=4,hidden_size=20,n_layers=3,n_features=40001,cell_type='biDRNN',prelayer=res(in_channels=4,n_layers=5,n_features=40001,init_ker_size=7,\
              block_type=block0,depth=2,zero_init=True,degrid=False,tail=False),n_pre=5,testrnn=False)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True) 
    
    
    train_loss_list=[]
    test_loss_list=[]
    count=0
    maxit=100
    itertime=0
    model=model.train()
    for count in range(maxit):
        if count==0 and quiet==False:
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
            
            output=model(data[0],True)
                #c_idx = c_index(output,torch.from_numpy(y_train),torch.from_numpy( censor_train))
            loss = MSELoss(reduction='sum')(output,data[1])#event=0,censored
            #train_loss_list.append(loss.cpu().data.numpy())
            t1=time.time()-t0
            #\print('forward takes {}s'.format(t1))
            train_loss_list.append(loss.cpu().data.numpy())
            if itertime%100==1 and quiet==False:
                print('train loss = {} at iter {}, lr = {}'.format(train_loss_list[-1],itertime,\
                      optimizer.param_groups[0]['lr']))
                if train_loss_list[-1]>1.1*train_loss_list[-2]:
                    optimizer.param_groups[0]['lr']=0.99*optimizer.param_groups[0]['lr']
           # t0=time.time()
            loss.backward()
            t1=time.time()-t0
            
            optimizer.step() 
            itertime+=1
            #print('backward takes {}s'.format(t1))
         
    model=model.eval()
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
