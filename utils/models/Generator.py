# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F



class EncDec(nn.Module):
    """docstring for MyLSTM"""
    def __init__(self,input_size, hidden_size, middle_dim, number_of_layers,dropout_value):
        super(EncDec, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.dropout = dropout_value
        self.layers = number_of_layers
        self.encoder = nn.LSTM(input_size,hidden_size,num_layers=number_of_layers,batch_first = True)
        self.decoder = nn.LSTM(input_size,hidden_size,num_layers=number_of_layers,batch_first = True)
        self.lin_in1 = nn.Linear(input_size,middle_dim)
        # self.lin_in2 = nn.Linear(input_size,middle_dim)
        self.lin_out = nn.Linear(self.hidden_dim,input_size)
        self.hidden = self.init_hidden()
         
    def init_hidden(self):
        
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.layers, 1, self.hidden_dim).cuda(),
                torch.zeros(self.layers, 1, self.hidden_dim).cuda())
         
    def forward(self,input):


        res = torch.rand(1,1)
        trav  = input.size(0)
        for b in range(trav):
            
            ### init the hiddens to zeros
            self.hidden = self.init_hidden()
            
            ### current input
            cur_inp = input[b].view(1,-1,4)
            
            ### first timestamp of current input will be given to decoder later
            decoder_input = cur_inp[0][0].view(1,1,4).clone()
            
            ### Get first desired number of timestamps of the data
            cur_inp = cur_inp[0][0:target_length].view(1,-1,4)
            
            
            #cur_inp = self.lin_in1(cur_inp) # TODO: CHECK dimensions and test the network without this torch.Size([1, 10, 12])
            

            out, self.hidden = self.encoder(cur_inp,self.hidden)


            ### To collect the output
            fakeDec = torch.ones(target_length,1, 4, dtype=torch.float, device=device)

            for di in range(target_length):

                decoder_output, self.hidden = self.decoder(
                    decoder_input, self.hidden)

                decoder_input = self.lin_out(decoder_output) 

                fakeDec[di] = decoder_input            
                
            if b == 0:
                fakeDec = fakeDec.view(1,-1,4)
                res = fakeDec
            else:
                fakeDec = fakeDec.view(1,-1,4)
                res = torch.cat((res,fakeDec),0)

        return res
