# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:16:55 2019

@author: Administrator
"""

from torch.nn import BatchNorm1d
a=torch.rand(2,4,8)
l=BatchNorm1d(4)
l(a).size()
n_padding=di=1
#if kernel_size==3
#then padding==dilation
cv=Conv1d(in_channels=4,out_channels=3,kernel_size=3,dilation=di,padding=n_padding)

cv=Conv1d(in_channels=4,out_channels=3,kernel_size=1,dilation=1,padding=0)
cv(a).size()

a=torch.rand(10**7)
a.element_size() * a.nelement()
2+5+2+1
40000000/(10**6)
a=torch.rand(10**9)
np.random.rand(1,2)
np.random.randint(0,2,[2,2])

data[0].size()
inputdata=torch.rand(3,100,4)
a1
a2=torch.from_numpy(np.flip(a1.numpy(),1).copy())
b1=biDRNN(n_input=4,n_hidden=10,n_layers=3 ,cell_type='LSTM', batch_first=True)

b1()




 

class biDRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False):
        super(biDRNN, self).__init__()
        self.drnn1=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False)
        self.drnn2=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False)
        
    def forward(self,input1):
        input2 = torch.from_numpy(np.flip(input1.numpy(),1).copy()) 
        out1=self.drnn1(input1)[0]
        out2=self.drnn2(input2)[0]
        return torch.cat([out1,out2],dim=-1)

rb=biDRNN(n_input=4,n_hidden=10,n_layers=3 ,cell_type='LSTM', batch_first=True) 
inputdata1=torch.rand(3,100,4)
assert rb(inputdata1).size()  == torch.Size([3,100,20])   
# 
# 
#dilated rnn
#cnn + dilated rnn
# 
# 

class testrnn(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_channels,hidden_size,n_layers,n_features,cell_type,prelayer,n_pre):
        #N input_channels,C channels,L length
        super(testrnn, self).__init__()
        factor=1.41
        if cell_type=='LSTM':
            self.rnn=LSTM(input_size=int(in_channels*factor**(n_pre)),hidden_size=hidden_size,num_layers=n_layers,bidirectional=True,batch_first=True)
        if cell_type=='GRU':
            self.rnn = GRU(input_size=int(in_channels*factor**(n_pre)),hidden_size=hidden_size,num_layers=n_layers,bidirectional=True,batch_first=True)
        if cell_type=='biDRNN':
            self.rnn = biDRNN(n_input=int(in_channels*factor**(n_pre)),n_hidden=hidden_size,n_layers=n_layers ,cell_type='LSTM', batch_first=True)
        if prelayer is None:
            init_ker_size = 7
            self.cn1=Conv1d(in_channels, in_channels, kernel_size=init_ker_size, stride=1,
                         padding=int((init_ker_size-1)/2), groups=1, bias=False, dilation=1)
            self.bn1=BatchNorm1d(in_channels)
            self.ac1=ReLU()
            self.prelayer=Sequential(*[self.cn1,self.bn1, self.ac1])
        else:
            self.prelayer=prelayer
        self.avdpool=AdaptiveAvgPool1d(1)
        self.fc = Linear( 2*hidden_size, 1)
    def forward(self, x ):
        x = self.prelayer(x)
        x = x.transpose(1,2)
        x = self.rnn(x)
        x = self.avdpool(x)
        print(x.size())
        x=x.view(x.size()[0],x.size()[1])
        x = self.fc(x)
        #x = self.lin2(x)
        return x

rnn1=testrnn(in_channels=4,hidden_size=10,n_layers=3,n_features=40001,cell_type='GRU',prelayer=res(in_channels=4,n_layers=4,n_features=40001,init_ker_size=7,\
          block_type=block0,depth=2,zero_init=True,degrid=False,tail=False),n_pre=4)

x1=next(iter(train_loader))
x1[0].transpose(1,2).size()
rnn1.rnn(rnn1.prelayer(x1[0]).transpose(1,2) )[0].size()
rnn1(x1[0]).size()
calc_num_par(rnn1)
