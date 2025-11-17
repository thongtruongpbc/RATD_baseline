import torch
import torch.nn as nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # TCN expects: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # Take the last output
        return self.linear(y[:, :, -1])
    
    def encode(self, x):
        # Encoding function for retrieval
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # Return the encoded representation
        return y[:, :, -1]  # Take the last time step 