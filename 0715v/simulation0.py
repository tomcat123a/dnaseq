# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:02:05 2019

@author: Administrator
"""

import torch
from seq import branch0,block0,block1,res,layer0,calc_num_par,SeqDataset,testcnn
from torch.utils.data import Dataset,DataLoader
from torch.nn import MSELoss
import time
import sys
import numpy as np
#from parsedna import totensor
#data
quiet=False   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

import time
t0=time.time()

seq=SeqDataset(tissue=0,exp_type=0,gen=False,chrom=6,pt_per=0,\
selected_patient_id=list(range(1)),len_per=1,seq_folder='/home/yilun/dna/seqdata/')
 
seqtest=SeqDataset(tissue=0,exp_type=0,gen=False,chrom=8,pt_per=0,\
selected_patient_id=list(range(1)),len_per=1,seq_folder='/home/yilun/dna/seqdata/')
params = {'batch_size': 4,
          'shuffle': True,
          'pin_memory':True} 


        
train_loader=DataLoader(seq,**params) 
test_loader=DataLoader(seqtest,**params)
 

print('loading genes from one patient takes {}s'.format(time.time()-t0))
#model=res(in_channels=4,n_layers=9,n_features=40001,init_ker_size=7,\
#          block_type=block0,depth=2,zero_init=True,degrid=False).to(device)
 
cnn_para_list=\
[{'in_channels':4,'n_layers':5,'n_features':100001,'depth':2,'block_type':block0,\
 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':7,'n_features':100001,'depth':2,'block_type':block0,\
 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':9,'n_features':100001,'depth':2,'block_type':block0,\
 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':5,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':7,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':9,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':5,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':7,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':9,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
{'in_channels':4,'n_layers':9,'n_features':100001,'depth':3,'block_type':block0,\
 'degrid':True,'testrnn':True,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True }]
#0-13 0,2,4,6,8,10,12 depth=2,1,3,5,7,9,11,13,depth=3,degrid=True,n_layers=3,4,5,6,7,8,9,testrnn=False
#14,15,16,n_layers=5,7,9,depth=2,testrnn=True
#17,18,19,n_layers=5,7,9,depth=2,testrnn=False,degrid=False
for ID in range(10):
    model=testcnn(**cnn_para_list[ID]).to(device)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001,amsgrad=True) 
    
    
    train_loss_list=[]
    test_loss_list=[]
    count=0
    max_epoch=1
    itertime=1
    model=model.train()
    for count in range(max_epoch):
        if count==0 and quiet==False:
            print('***********************')
            print('case {}'.format(ID))
            print('***********************')
            print('batch size{}'.format(params['batch_size']))
            print('num of pars in model={}'.format(calc_num_par(model)))
            print('max epoch = {}'.format(max_epoch))
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
            #\
            train_loss_list.append(loss.cpu().data.numpy())
            if itertime%20==0 and quiet==False:
                print('train loss = {} at iter {}, lr = {}'.format(train_loss_list[-1],itertime,\
                      optimizer.param_groups[0]['lr']))
                print('forward takes {}s'.format(t1))
                
           # t0=time.time()
            loss.backward()
            t1=time.time()-t0
            
            optimizer.step() 
            itertime+=1
            #print('backward takes {}s'.format(t1))
         
    model=model.eval()
    pred=[]
    real=[]
    for data in test_loader:
        data[0] = data[0].to(device)
        data[1] = data[1].to(device) 
        output = model(data[0])
        pred.append(output.cpu().data.numpy().reshape(-1))
        real.append(data[1].cpu().data.numpy())
    cor = np.corrcoef(np.concatenate(pred),np.concatenate(real))[0,1]
        
    print('final R square = {}, corrcoef = {}'.format(cor**2, cor))
