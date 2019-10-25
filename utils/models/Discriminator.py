import torch
import torch.nn as nn



class Discriminator(nn.Module):
  ## As arguments, input_size(how many items in one timestamp), hidden_size, number of lstm layers and dropout value should be given.
    def __init__(self, input_size, hidden_size, number_of_layers,dropout_value):
        super(Discriminator, self).__init__()
        self.layers = number_of_layers
        self.hidden_dim = hidden_size
        self.input_size = input_size
        self.dropout = dropout_value
        self.lstm1 = nn.LSTM(input_size,hidden_size,num_layers=number_of_layers,batch_first=True)

        self.lin1 = nn.Linear(self.hidden_dim,1)

        self.hidden = self.init_hidden()
         
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.layers, 1, self.hidden_dim).cuda(),
                torch.zeros(self.layers, 1, self.hidden_dim).cuda())
    def forward(self, input):
        
        ### Results will be kept in this variable
        res = torch.rand(1,1)
        ### Batch size
        trav  = input.size(0)
        for b in range(trav):
          
            #### Init hidden to zeros
            self.hidden = self.init_hidden()
            
            #### For every timestamp that we are interested do the calculations.
              
            disc_input = input[b][0:target_length].view(1,-1,self.input_size)
                
                
                
            disc_output, self.hidden = self.lstm1(disc_input, self.hidden)

            #### First element of the hidden is the hidden outputs, second one is the memory (h,c)
            out = self.lin1(self.hidden[0])
            
            #### This sigmoid should remain closed as long as we are using BCELossLogit
            #out = torch.sigmoid(out)
            
            #### For first sample of the batch result should be initiliazed with it
            if b == 0:
                res = out
            #### For the rest it will get concat to previously initiliazed "res" variable
            else:
                res = torch.cat((res,out),0)
        return res