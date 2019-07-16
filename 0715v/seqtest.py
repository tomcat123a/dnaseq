# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:29:59 2019

@author: Administrator
"""
import torch
from seq import branch0,block0,block1,res,layer0,testcnn,SeqDataset,calc_num_par
from torch.utils.data import Dataset,DataLoader 
from torch.nn import MSELoss   
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

inputx=torch.rand(5,4,100)#ncl
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

inputx=torch.rand(5,4,100)
mod=testcnn(in_channels=4,n_layers=2,n_features=100,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=2,n_features=100,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=True,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])
inputx=torch.rand(5,4,40001)
mod=testcnn(in_channels=4,n_layers=9,n_features=40001,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=9,n_features=40001,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=True,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=5,n_features=40001,block_type=block0,depth=3,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=20,n_layers_rnn=4,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=5,n_features=40001,block_type=block0,depth=3,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=20,n_layers_rnn=4,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=6,n_features=40001,block_type=block0,depth=2,degrid=True,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=6,n_features=40001,block_type=block0,depth=2,degrid=True,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=2,n_features=40001,block_type=block0,depth=2,degrid=False,cell_type='GRU',\
            testrnn=False,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=2,n_features=100,block_type=block0,depth=2,degrid=False,cell_type='GRU',\
            testrnn=False,hidden_size=10,n_layers_rnn=2,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])



mod=testcnn(in_channels=4,n_layers=3,n_features=40001,block_type=block0,depth=2,degrid=False,cell_type='LSTM',\
            testrnn=False,hidden_size=20,n_layers_rnn=4,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])

mod=testcnn(in_channels=4,n_layers=8,n_features=40001,block_type=block0,depth=2,degrid=False,cell_type='LSTM',\
            testrnn=True,hidden_size=20,n_layers_rnn=4,rnn_gpu=False)
mod(inputx).size()==torch.Size([inputx.size()[0],1])
print('basic test done')
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
dt=iter(train_loader)
def testfunc(mod,dt=dt,device=None):
    print('#par={}'.format(calc_num_par(mod)))
    
        
    tx = next(dt)[0]
    print(tx.size())
    if device is not None:
        mod=mod.to(device)
        tx=tx.to(device)
    t0=time.time()
    a = mod(tx)
    print(a.size())
    print('forward time={}'.format(time.time()-t0))
    t0=time.time()
    torch.pow(a.sum(),2).backward()
    print('backward time={}'.format(time.time()-t0))
    

    
    
mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=False)
testfunc(mod) 
mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=False)
testfunc(mod) 
mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=3,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=False)
testfunc(mod) 
mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=3,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=False)
testfunc(mod) 
mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=3,degrid=True,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=False)
testfunc(mod) 

mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=3,degrid=True,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=False)
testfunc(mod) 

mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=2,degrid=True,cell_type='biDRNN',\
            testrnn=True,hidden_size=10,n_layers_rnn=3,rnn_gpu=False)
testfunc(mod) 

mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=3,degrid=True,cell_type='biDRNN',\
            testrnn=True,hidden_size=10,n_layers_rnn=3,rnn_gpu=False)
testfunc(mod) 

print('cpu version test finished')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=True)

testfunc(mod,device=device) 
mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=2,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=True)
testfunc(mod,device=device) 
mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=3,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=True)
testfunc(mod,device=device) 
mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=3,degrid=False,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=True)
testfunc(mod,device=device) 
mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=3,degrid=True,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=True)
testfunc(mod,device=device) 

mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=3,degrid=True,cell_type='biDRNN',\
            testrnn=False,hidden_size=10,n_layers_rnn=4,rnn_gpu=True)
testfunc(mod,device=device) 
mod=testcnn(in_channels=4,n_layers=5,n_features=100001,block_type=block0,depth=2,degrid=True,cell_type='biDRNN',\
            testrnn=True,hidden_size=10,n_layers_rnn=3,rnn_gpu=True)
testfunc(mod,device=device) 

mod=testcnn(in_channels=4,n_layers=9,n_features=100001,block_type=block0,depth=3,degrid=True,cell_type='biDRNN',\
            testrnn=True,hidden_size=10,n_layers_rnn=3,rnn_gpu=True)
testfunc(mod,device=device) 
