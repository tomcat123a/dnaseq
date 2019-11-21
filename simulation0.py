# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:02:05 2019

@author: Administrator
"""

import torch
from seq import branch0,block0,block1,res,layer0,calc_num_par,SeqDatasetpt,testcnn,testmlp,testmlp_chunk,testlinear,test_rnn,Newloader,print_info,embed
from seq import cnn59,NewGenewiseloader,generate_sampler,NewGenewiseloader_chunk,test_mix
from seq import Beluga
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from seq import getfeature_xg
from torch.nn import MSELoss,L1Loss
import time
import os
import numpy as np

import pandas as pd
 
torch.manual_seed(2)    
np.random.seed(1)    
#from parsedna import totensor
#data
quiet=True   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
#device ='cpu'        
 
t0=time.time()
#params = {'batch_size':64,'epochs':1000,'len_percent':.105,'lr':5e-5,'pin_memory':True,'detail':False,'interval':1}

params = {'batch_size':90,'epochs':100,'len_percent':.3,'lr':1e-3,\
          'pin_memory':False,'detail':False,'interval':1,'chunk':2}

#params = {'batch_size':256,'epochs':1000,'len_percent':.2,'lr':5e-5,'pin_memory':True,'detail':False,'interval':5}
  

#model=res(in_channels=4,n_layers=9,n_features=40001,init_ker_size=7,\
#          block_type=block0,depth=2,zero_init=True,degrid=False).to(device)
DATA_TYPE=1
MODEL_TYPE=1
 #1 cnn+dnn,4,cnn59
'''
SEED_TYPE=0
TISSUE_TYPE=1
'''
print_count=0
FINAL_STORE_LIST=[]
TISSUE_TYPE=2
for REV_TYPE in [0]:#reverse_type=4,augumented in training,average in val and test 
    #reverse_type=1,,augumented in training,using correct seq in val and test 
    print('start building loader at REV_TYPE{}'.format(REV_TYPE))
     #0,raw,1,nor_batch,2,unnor_batch
     
    seqtrain=NewGenewiseloader(tissue=TISSUE_TYPE,chrom=list(range(1,23)),\
                seq_folder='/media/yilun/Elements/RefSeQ/DeepSeQ2/',\
                exp_folder='/media/yilun/Elements/DeepSeQ/expr/avg_0/' ,\
                in_mem=False,len_per=params['len_percent'],rev_train=REV_TYPE\
                ,start=0.42,end=0.42)
    '''
    seqtrain=NewGenewiseloader_chunk(tissue=TISSUE_TYPE,chrom=list(range(1,23)),\
                seq_folder='/media/yilun/Elements/RefSeQ/DeepSeQ2/',\
                exp_folder='/media/yilun/Elements/DeepSeQ/expr/avg_0/' ,\
                in_mem=True,len_per=params['len_percent'],rev_train=REV_TYPE,chunk =params['chunk'] )
    ''' 
    for SEED_TYPE in range(1):
        for MODEL_TYPE in [12]:
        

            if DATA_TYPE==0:
                params = {'batch_size':128,'epochs':100,'len_percent':.05,'lr':5e-5,'pin_memory':True,'detail':False,'interval':5}
                seqtrain=Newloader(tissue=1,exp_type=0,chrom=[1,2],section=[0,1,2,3,4,5,6,7,8],pt_per=1,\
                selected_patient_id=range(0,20),len_per=params['len_percent'],seq_folder='/media/yilun/Elements/DeepSeQ/unnormalized/',\
                in_mem=True)
                train_loader=DataLoader(seqtrain,batch_size=params['batch_size'],pin_memory=params['pin_memory'])
                seqtest=Newloader(tissue=1,exp_type=0,chrom=[1,2],section=[0,1,2,3,4,5,6,7,8],pt_per=1,\
                selected_patient_id=list(range(30,40)),len_per=params['len_percent'],seq_folder='/media/yilun/Elements/DeepSeQ/unnormalized/',in_mem=True )
                #params['batch_size']
                test_loader=DataLoader(seqtest,batch_size=params['batch_size'],pin_memory=params['pin_memory'])
                
                print(seqtrain.__len__())
                print('building dataloader  takes {}s'.format(time.time()-t0))
            else:
                print('dataloader length'+str(seqtrain.__len__()))
                print('testing mod{} tissue{} rev_type{} seed{}'.format(MODEL_TYPE,1,REV_TYPE,SEED_TYPE))#len_percent=.105 usecache=1,2:len_percent=.2,usecache=3,4
                
                if seqtrain.rev_train==0:
                    train_indices,val_indices,test_indices=generate_sampler(seqtrain.__len__(),\
                    0.8,0.1,SEED_TYPE) 
                    train_loader=DataLoader(seqtrain,batch_size=params['batch_size'],\
                    pin_memory=params['pin_memory'],sampler=SubsetRandomSampler(train_indices))
                    val_loader=DataLoader(seqtrain,batch_size=params['batch_size'],\
                    pin_memory=params['pin_memory'],sampler=SubsetRandomSampler(val_indices))
                    test_loader=DataLoader(seqtrain,batch_size=params['batch_size'],\
                    pin_memory=params['pin_memory'],sampler=SubsetRandomSampler(test_indices))
                if seqtrain.rev_train==1 or seqtrain.rev_train==4:
                    train_indices,val_indices,test_indices=generate_sampler(int(seqtrain.__len__()/2),\
                    0.8,0.1,SEED_TYPE) 
                    train_indices=np.concatenate((2*train_indices,1+2*train_indices),0)
                    train_loader=DataLoader(seqtrain,batch_size=params['batch_size'],\
                    pin_memory=params['pin_memory'],sampler=SubsetRandomSampler(train_indices))
                    val_loader=DataLoader(seqtrain,batch_size=params['batch_size'],\
                    pin_memory=params['pin_memory'],sampler=SubsetRandomSampler(2*val_indices),shuffle=False)
                    
                    test_loader=DataLoader(seqtrain,batch_size=params['batch_size'],\
                    pin_memory=params['pin_memory'],sampler=SubsetRandomSampler(2*test_indices),shuffle=False)
                 
                
                if quiet==True:
                    print(seqtrain.__len__())
                    print('building dataloader  takes {}s'.format(time.time()-t0))
            
            if  MODEL_TYPE==10:
                cnn_para_list=\
                    [{'L':seqtrain.dna_len,'in_channels':4,'out_channels':12,'ker_size':[3,5,7,9,11,13],'stride':[1],\
                      'dilation':[1,2,3,4,6,8],\
                      'poolsize':[100] ,'filter_type':'avgpool'}\
                    ]
            
            
            
            #chunk version
            if  MODEL_TYPE==9:
            #cnn+dnn 
                '''
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']*params['chunk']),\
                  'init_ker_size':7,'stride':[1,1 ],\
                  'channels_list':[4,100],'dnn_list':[3,1],'di':3 ,'res_on':True,'poolsize':[30],\
                  'use_rnn':False,'chunk_size':params['chunk'],'chunk_id':[0,1,2,3,4]}
                ]
                '''
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']*params['chunk']),\
                  'init_ker_size':7,'stride':[1,1 ],\
                  'channels_list':[4,100,100 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[800,200 ],\
                  'use_rnn':False,'chunk_size':params['chunk'],'chunk_id':[0,1]}
                ]
            
            if  MODEL_TYPE==4:
            #cnn+dnn 
                cnn_para_list=\
                ['null'] 
            
            if  MODEL_TYPE==12:#expecto only forward
            #cnn+dnn 
                cnn_para_list=\
                ['null'] 
            
            if  MODEL_TYPE==8:
            #cnn+dnn 
                cnn_para_list=\
                [{'dnn_list':[1]}] 
            
            if  MODEL_TYPE==0:
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']) ,'dnn_list':[1000,1] }
                ]
            
            if  MODEL_TYPE==1:
            #cnn+dnn 
                if REV_TYPE in [0,1,4]:
                    cnn_para_list=\
                    [{'n_features':seqtrain.dna_len,'ker_size':[7]+[3]*27 ,'stride':[2,1,1,1]+[3,1,1]*7,\
                      'channels_list':[4,100,100,100,140, 140,140,200,200,200,300,300,300,450,450,450,600,600,600,800,800,800,900,900,900 ],'dnn_list':[2,1],'di':[1]*28,'res_on':True,\
                      'poolsize':[],'use_rnn':False }\
                    ]
                    '''
                    [{'n_features':1+int(2*50000*params['len_percent']),'ker_size':[5]*8,'stride':[1]*8,\
                      'channels_list':[4,200,200,200, 200,200 ],'dnn_list':[4,1],'di':[1,2,4,8,16  ] ,'res_on':True,\
                      'poolsize':[10000,1000,200, 80,80 ],'use_rnn':False }\
                    ]
                    '''
                    ''' 
                    [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                      'channels_list':[4,100,100,180 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[900,200,80   ],'use_rnn':True}
                    ]
                    '''
                    '''[{'n_features':seqtrain.dna_len,'ker_size':[7,3,3,3,3,3,3 ,3  ],'stride':[1,1,1,2,2,2,4,4   ] ,\
                      'channels_list':[4,100,100,100,200,200,300,400 ,  ],'dnn_list':[1],'di':[1,1,2,4,4,4,4,4   ] ,'res_on':True,\
                      'poolsize':[ ],'use_rnn':False }
                    ]
                    '''
                      #seq length 5000
                    
                if REV_TYPE in [2]:
                    cnn_para_list=\
                    [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                      'channels_list':[4,100,100,180 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[1800,400,160   ],'use_rnn':True}
                    ]
                if REV_TYPE in [3]:
                    cnn_para_list=\
                    [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                      'channels_list':[8,100,100,180 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[900,200,80   ],'use_rnn':True}
                    ]
                '''
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                  'channels_list':[4,100,120,200 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[1000,200,80  ]}
                ]
                '''
            if  MODEL_TYPE==5:
            #cnn+dnn 
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                  'channels_list':[4,150,150,300 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[900,200,80   ],'use_rnn':False}
                ]
                '''
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                  'channels_list':[4,100,120,200 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[1000,200,80  ]}
                ]
                '''
            if  MODEL_TYPE==6:
            #cnn+dnn 
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1  ],\
                      'channels_list':[4,100,100,180    ],'dnn_list':[2,1],'di':1 ,'res_on':True,\
                      'poolsize':[1500,400,100 ],'use_rnn':False,'extra':True}
                ]
                '''
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                  'channels_list':[4,100,120,200 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[1000,200,80  ]}
                ]
                '''
            if  MODEL_TYPE==7:
            #cnn+dnn 
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                  'channels_list':[4,100,100,180 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[900,200,80   ],'use_rnn':True}
                ]
                '''
                [{'n_features':1+int(2*50000*params['len_percent']),'init_ker_size':7,'stride':[1,1,1,1 ],\
                  'channels_list':[4,100,120,200 ],'dnn_list':[2,1],'di':2 ,'res_on':True,'poolsize':[1000,200,80  ]}
                ]
                '''
            if  MODEL_TYPE==2:
                #RNN+DNN
                #width 50 ok
                cnn_para_list=\
                [{'n_features':1+int(2*50000*params['len_percent']),'width':20,'out_channels':10,'hidden':6,'n_layers':2,'dnn_list':[1] }
                ] 
            
              
            if  MODEL_TYPE==3: 
            
                cnn_para_list=\
                [{'in_channels':4,'n_layers':1,'n_features':1+int(2*50000*params['len_percent']),'depth':2,'block_type':block0,\
                 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True,'res_on':True },
                {'in_channels':4,'n_layers':7,'n_features':1+int(2*50000*params['len_percent']),'depth':2,'block_type':block0,
                 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':9,'n_features':1+int(2*50000*params['len_percent']),'depth':2,'block_type':block0,
                 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':5,'n_features':1+int(2*50000*params['len_percent']),'depth':3,'block_type':block0,
                 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':7,'n_features':1+int(2*50000*params['len_percent']),'depth':3,'block_type':block0,
                 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':9,'n_features':1+int(2*50000*params['len_percent']),'depth':3,'block_type':block0,
                 'degrid':False,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':5,'n_features':1+int(2*50000*params['len_percent']),'depth':3,'block_type':block0,
                 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':7,'n_features':1+int(2*50000*params['len_percent']),'depth':3,'block_type':block0,
                 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':9,'n_features':1+int(2*50000*params['len_percent']),'depth':3,'block_type':block0,
                 'degrid':True,'testrnn':False,'cell_type':'biDRNN','hidden_size':10,'n_layers_rnn':3,'rnn_gpu':True },
                {'in_channels':4,'n_layers':5,'n_features':1+int(2*50000*params['len_percent']),'depth':2,'block_type':block0,
                 'degrid':True,'testrnn':True,'cell_type':'biDRNN','hidden_size':5,'n_layers_rnn':3,'rnn_gpu':True }]
            
            #0-13 0,2,4,6,8,10,12 depth=2,1,3,5,7,9,11,13,depth=3,degrid=True,n_layers=3,4,5,6,7,8,9,testrnn=False
            #14,15,16,n_layers=5,7,9,depth=2,testrnn=True
            #17,18,19,n_layers=5,7,9,depth=2,testrnn=False,degrid=False
            '''
            def init_all(model, init_func, *params, **kwargs):
                for p in model.parameters():
                    init_func(p, *params, **kwargs)
            '''
            for ID in range(1):
                if print_count==0:
                    print(cnn_para_list[ID])
                    print_count=print_count+1
                if MODEL_TYPE==0:
                    model=testlinear(**cnn_para_list[ID]).to(device)
                if MODEL_TYPE in [1,5,6,7]:
                    
                    model=testmlp(**cnn_para_list[ID]).to(device)
                #
                if MODEL_TYPE==8:#embedding
                    model=embed(**cnn_para_list[ID]).to(device)
                if MODEL_TYPE==2:
                    model=test_rnn(**cnn_para_list[ID]).to(device)
                if MODEL_TYPE==3:    
                    model=testcnn(**cnn_para_list[ID]).to(device)
                if MODEL_TYPE==9:    
                    model=testmlp_chunk(**cnn_para_list[ID]).to(device)
                if MODEL_TYPE==4:    
                    model=cnn59( ).to(device)
                if MODEL_TYPE==10:    
                    model=cnn59( ).to(device)
                    
                if MODEL_TYPE==12:
                    model = Beluga()
                    #model.load_state_dict(torch.load('./resources/deepsea.beluga.pth'))
                    model.load_state_dict(torch.load('/home/yilun/DNA_research_201911/expector_dir/ExPecto/resources/deepsea.beluga.pth'))
                    model.eval()
                    model.to(device)
                    print('beluga loaded!')
                    #special treatment for beluga
                    getfeature_xg(model,train_loader,device,seqtrain,1)
                    import xgboost
                    getfeature_xg(model,val_loader,device,seqtrain,2) 
                    getfeature_xg(model,test_loader,device,seqtrain,3) 
                    '''
                    np.save('train_x',train_x)
                    np.save('train_y',train_y)
                    np.save('pred_x',pred_x)
                    np.save('pred_y',pred_y)
                    train_x=train_x.reshape(train_x.shape[0],-1)
                    pred_x=pred_x.reshape(pred_x.shape[0],-1)
                    print('xgb starts')
                    xgb = xgboost.XGBRegressor( random_state=0)
                    xgb.fit(train_x,train_y)
                    print('xgb fitted')
                    
                    yhat=xgb.predict(pred_x)
                    print(yhat.shape)
                    np.save('yhat',yhat)
                    np.save('realy',pred_y)
                    print(pred_y.shape)
                    cor=np.corrcoef(np.concatenate(yhat),np.concatenate(pred_y))[0,1]
                    print('cor xgboost {},R2 {}'.format(cor,cor**2))
                    '''
                    break 
                optimizer=torch.optim.Adam(model.parameters(), lr=params['lr'],amsgrad=True) 
                #optimizer=torch.optim.Adam(nn.ParameterList(model.parameters()), lr=params['lr'],amsgrad=True) 
                #optimizer=torch.optim.SGD(model.parameters(),lr=params['lr']) 
                
                #optimizer=torch.optim.RMSprop(model.parameters(), lr=params['lr'] ) 
                
                train_loss_list=[]
                test_loss_list=[]
                count=0
                max_epoch=params['epochs']
                model=model.eval()
                #print('pre-training evaluation before epoch 1')
                 
                #print_info(train_loader,model,device,'before training data')
                 
                #print_info(val_loader,model,device,' val data')
                
                model=model.train()
                
                
                
                prevcount=0
                if quiet==False or quiet==True:
                    #print('***********************')
                    #print('case {}'.format(ID))
                    #print('***********************')
                    #print('batch size{}'.format(params['batch_size']))
                    print('num of pars in model={}'.format(calc_num_par(model)))
                    #print('max epoch = {}'.format(max_epoch))
                    #print( 'param:'+str(cnn_para_list[ID]))
                if train_loader.__len__()==1:
                    print('\n')
                    print('Full batch mode enabled.')
                    print('\n')
                t30=time.time()
                used_up=False
                prev_valR2=0
                prev_testR2=0
                for count in range(1,max_epoch+1):
                    itertime=0
                    for datax,datay in train_loader :
                        t0=time.time()      
                        model=model.train()
                        optimizer.zero_grad()
                        
                        datax = datax.to(device)
                        datay = datay.to(device)      
                        t_load=time.time()-t0
                         
                        output=model(datax)
                        if output.dim()>1:
                            output= output.squeeze(-1)
                        if datay.dim()>1:
                            datay=  datay.squeeze(-1)
                         
                        assert datay.size()==output.size()
                        loss = MSELoss(reduction='mean')(output,datay) 
                        '''
                        print('output size')
                        print(output.cpu().size())
                        print(output.cpu().data.numpy())
                        print('datay size')
                        print(datay.cpu().size())
                        print(datay.cpu().data.numpy())
                        print('loss size')
                        print(loss.cpu().size())
                        print(loss.cpu().data.numpy())
                        '''
                        t1=time.time()-t0
                       
                        train_loss_list.append( float( loss.cpu().data.numpy() ))
                        
                        t0=time.time()
                        
                        loss.backward()
                        optimizer.step() 
               
                        t2=time.time()-t0
                       
                              
                        itertime+=1
                        ##haha
                        #pred,real=print_info(loader=val_loader,mod=model,device=device,\
                        #                     text='intermediate VALIDATION data',quiet=False,pair=(REV_TYPE==4))
                        if itertime%10==1:
                            print_info(loader=test_loader,mod=model,device=device,\
                                                 text='intermediate test data at epoch{} iter{}'.format(count,itertime),quiet=False,pair=(REV_TYPE==4))
                        
                        if   quiet==False and prevcount==count-1 :
                            #print('quiet')
                            prevcount = prevcount+1
                            t3=time.time()-t30
            
                            if count > 1 and params['detail']: 
                                print('avd_batch_mse = {:.4f} progression% {:.2f}%,lr = {:.7f},batch_load_to_device_time = {:.4f}s'.\
                                      format(train_loss_list[-1],\
                                      (count)*100/(params['epochs']),\
                                      optimizer.param_groups[0]['lr'],t_load))
                             
                            if params['detail']:
                                #print(output[0].cpu().data.numpy())
                                print('last 6 pred vs real \n {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.\
                                      format(output[0].cpu().data.numpy() ,output[1].cpu().data.numpy() ,output[2].cpu().data.numpy() ,\
                                             output[-6].cpu().data.numpy() ,\
                                      output[-5].cpu().data.numpy() ,\
                                      output[-4].cpu().data.numpy() ,output[-3].cpu().data.numpy() ,\
                                      output[-2].cpu().data.numpy() ,\
                                      output[-1].cpu().data.numpy() ,datay[0].cpu().data.numpy(),datay[1].cpu().data.numpy(),\
                                      datay[2].cpu().data.numpy(),datay[3].cpu().data.numpy(),datay[-6].cpu().data.numpy(),datay[-5].cpu().data.numpy(),\
                                      datay[-4].cpu().data.numpy(),datay[-3].cpu().data.numpy(),datay[-2].cpu().data.numpy(),datay[-1].cpu().data.numpy())) 
                            if count >1:     
                                print('forward takes {:.5f}s back takes {:.5f}s remaining time {:.3f}'.\
                                      format(t1,t2,t3/((count)/(params['epochs']))-t3) )
                            else:
                                print('forward takes {:.5f}s back takes {:.5f}s  '.\
                                      format(t1,t2) )
                            
                        
                            
                   
                    if count%params['interval']==params['interval']-1 :
                        #print('intermediate evaluation starts after epoch {} finished!'.format(count))
                        model=model.eval()
                        #print_info(train_loader,model,device,'intermediate training data')
                        pred,real=print_info(loader=val_loader,mod=model,device=device,\
                                             text='intermediate VALIDATION data',quiet=False,pair=(REV_TYPE==4))
                        cor = np.corrcoef(np.concatenate(pred),np.concatenate(real))[0,1]
                        current_valR2=cor**2
                        
                        if count > 1:
                            if prev_valR2>=current_valR2:
                                optimizer['lr']=0.1*optimizer['lr']
                                print('~~~~~~~~~~~~~~~~~~~~~~~~~~lr shrunk!')
                                print('\n')
                                if optimizer['lr']==params['lr']*0.0001:
                                    print('Finished! TestR2={} '.format(prev_testR2) )
                                     
                                    FINAL_STORE_LIST.append(prev_testR2)
                                    break
                        pred,real=print_info(loader=test_loader,mod=model,device=device,\
                                             text='intermediate test data',quiet=False,pair=(REV_TYPE==4))
                        cor = np.corrcoef(np.concatenate(pred),np.concatenate(real))[0,1]
                        prev_testR2=cor**2
                        prev_valR2=current_valR2    
                        pd.DataFrame(pred).to_csv("pred.csv",index=False)
                        pd.DataFrame(real).to_csv("real.csv",index=False)
                        if count==2:
                            pd.DataFrame(real).to_csv("real.csv",index=False)
                        model=model.train()
                if count==max_epoch:
                    used_up=True
                
                #torch.save(model, 'mod_final.pkl')
                
                
                model=model.eval()
                
                
                 
                
                #model=model.eval()
                if used_up==True:
                    #print('Doing training evaluation' )
                    #print_info(train_loader,model,device,'final training data')
                    print('Doing test evaluation' )
                    pred,real=print_info(test_loader,model,device,'final test data')
                    cor = np.corrcoef(np.concatenate(pred),np.concatenate(real))[0,1]
                    FINAL_STORE_LIST.append(cor**2)
                 
                if DATA_TYPE<1:
                    n_genes=seqtest.exp_shape[0]
                    n_patients=seqtest.exp_shape[1]
                    print('Gene specific R^2:')
                    print('#genes {},#patients {}'.format(n_genes,n_patients))
                    predmat=np.concatenate(pred).reshape((n_genes,n_patients))
                    #torch.save(predmat,'pred{}_100.pt'.format(N_TRAIN))
                    pd.DataFrame(predmat).to_csv("pred.csv",index=False)
                    predmat0=predmat.copy()
                    
                    for j in range(n_patients):
                        predmat0[:,j]=predmat[:,j]-predmat[:,j].mean()
                        predmat0[:,j]=predmat0[:,j]/np.sqrt(sum(predmat0[:,j]**2))
                    
                    #print(predmat.mean(axis=0))
                    
                    realmat=np.concatenate(real).reshape((n_genes,n_patients))
                    #torch.save(realmat,'real{}_100.pt'.format(N_TRAIN))
                    pd.DataFrame(realmat).to_csv("real.csv",index=False)
                    #torch.save(predmat0,'pred{}_100_zscore.pt'.format(N_TRAIN))
                    #predmat[0] is the predicted first gene's expression for all patients
                    result=np.ones((n_genes,2))
                    result_zscore=np.ones((n_genes,2))
                    for j in range(n_genes):
                        if 10*(j/n_genes)==np.floor(10*(j/n_genes)):
                            print('{}%progression of test data cor fininshed'.format(100*np.floor((j/n_genes))))
                        predrow =predmat[j]
                        realrow =realmat[j]
                        predrow_zscore =predmat0[j]
                        
                        #print('pred')
                        #print(predrow)
                        #print('real')
                       # print(realrow)
                        cor = np.corrcoef(predrow,realrow)[0,1]
                        result[j,0]=cor
                        result[j,1]=cor**2
                        cor = np.corrcoef(predrow_zscore,realrow)[0,1]
                        result_zscore[j,0]=cor
                        result_zscore[j,1]=cor**2
                        
                     
                    print('gene_specific corcoef and R^2.')
                    print(result)
                    os.system('Rscript HistCor.R  -r ./real.csv -p ./pred.csv ')
                    #torch.save(result,'result.pt')
                    #torch.save(result_zscore,'result_zscore.pt')
                    #result=result[~np.isnan(result[:,0])]
                    #print('top 20{},bot 20()'np.sort( abs(result[:,1] ) )
                 
                del train_loader
                del val_loader
                del test_loader
                del model
                del datax
                del datay
                    
     
pd.DataFrame(FINAL_STORE_LIST).to_csv('R2_list.csv')   
