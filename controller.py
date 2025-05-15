import torch
from torch import nn
import torch.autograd as autograd








class LSTMController(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)

    def create_new_state(self, batch_size):

        lstm_h = autograd.Variable(torch.zeros(self.num_layers, batch_size, self.num_outputs))
        lstm_c = autograd.Variable(torch.zeros(self.num_layers, batch_size, self.num_outputs))
        
        return lstm_h, lstm_c

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state, prev_reads=None, class_vector=None, seq=1):
        for i in range(seq):
            x = x.unsqueeze(0)
            outp, state = self.lstm(x, prev_state)
            prev_state=state

        return outp[-1], state