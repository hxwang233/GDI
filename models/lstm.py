import torch
import torch.nn as nn
import torch.nn.init as init

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_class, bidirectional, output_activation=torch.log_softmax):
        super(RNN, self).__init__()
        #(input_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, bidirectional = bidirectional, batch_first = True)
        num_directions = 1
        if bidirectional:
            num_directions = 2
        self.out = nn.Linear(hidden_size * num_directions, n_class)  # (fcn_hidden_size, n_class)
        self.output_activation = output_activation
        return

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.out(x[:,-1,:])
        out = self.output_activation(out, dim=1)
        return out