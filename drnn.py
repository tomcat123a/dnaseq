# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:29:08 2019

@author: Administrator
"""

import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool1d,Linear,BatchNorm1d,Conv1d


use_cuda = torch.cuda.is_available()


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            
            if i == 0:
                c = cell(n_input,  n_hidden  , dropout=dropout)
            else:
                c = cell(n_hidden,  n_hidden , dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])

            outputs.append(inputs[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden
        
        


class biDRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers , dropout=0, cell_type='GRU', batch_first=False):
        super(biDRNN, self).__init__()
        self.drnn1=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type=cell_type, batch_first=False)
        self.drnn2=DRNN(n_input, n_hidden, n_layers , dropout=0, cell_type=cell_type, batch_first=False)
        
    def forward(self,input1):
        #input1 is (N,L,C) N batch size ,L seq length, C #channels
        input2 = torch.flip(input1,dims=[1])
        out1=self.drnn1(input1)[0]
        out2=self.drnn2(input2)[0]
         
        return torch.cat([out1,out2],dim=-1)



class testrnn(torch.nn.Module):
    
    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    def __init__(self,in_channels,hidden_size,n_layers,n_features,cell_type,prelayer,n_pre):
        #N input_channels,C channels,L length
        super(testrnn, self).__init__()
        factor=1.41
        self.cell_type=cell_type
        
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
        self.avdpool_cnn=AdaptiveAvgPool1d(1)
        self.fc_cnn = Linear(int(factor**n_pre*in_channels), 1)
        self.avdpool_rnn=AdaptiveAvgPool1d(1)
        self.fc_rnn = Linear( 2*hidden_size, 1)
    def forward(self, x , rnn=True ):
        x = self.prelayer(x)
         
        x1 = self.avdpool_cnn(x)
        x1 = x1.squeeze(dim=-1)
        x1 = self.fc_cnn(x1)
        
        if rnn==True:
            #(N,C,L) to (N,L,C)
            x = x.transpose(1,2)
            #since rnn is batch first
            if self.cell_type!='biDRNN':
                x = self.rnn(x)[0] 
            else:
                x = self.rnn(x)
            x = x.transpose(1,2)
            x = self.avdpool_rnn(x)
            x = x.squeeze(dim=-1)
            x = self.fc_rnn(x)
        #x = self.lin2(x)
            return x+x1
        else:
            return x1
        
        
