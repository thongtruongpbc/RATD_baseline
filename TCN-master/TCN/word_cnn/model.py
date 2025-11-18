import torch
import torch.nn as nn
from TCN.tcn import TemporalConvNet
class TCNEncoder(nn.Module):
    def __init__(self, input_size, latent_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], latent_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)            # (batch, input_size, seq_len)
        x = x.float()
        y = self.tcn(x)                  # (batch, channels, seq_len)
        z = self.linear(y[:, :, -1])     # latent embedding (batch, latent_size)
        return z

class TCNDecoder(nn.Module):
    def __init__(self, latent_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNDecoder, self).__init__()
        self.linear = nn.Linear(latent_size, num_channels[-1])
        self.tcn = TemporalConvNet(num_channels[-1], num_channels, kernel_size=kernel_size, dropout=dropout)
        self.output_linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.output_linear.weight.data.normal_(0, 0.01)

    def forward(self, z, seq_len):
        # z: (batch, latent_size)
        x = self.linear(z)                # (batch, num_channels[-1])
        x = x.unsqueeze(-1).repeat(1, 1, seq_len)  # (batch, channels, seq_len)
        x = x.float()
        y = self.tcn(x)                   # (batch, channels, seq_len)
        x_hat = self.output_linear(y.transpose(1, 2))  # (batch, seq_len, output_size)
        return x_hat

class TCN_AE(nn.Module):
    def __init__(self, input_size, latent_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN_AE, self).__init__()
        self.encoder = TCNEncoder(input_size, latent_size, num_channels, kernel_size, dropout)
        self.decoder = TCNDecoder(latent_size, input_size, num_channels, kernel_size, dropout)
    def forward(self, x):
        seq_len = x.size(1)
        x = x.float()
        z = self.encoder(x)               # latent embedding
        x_hat = self.decoder(z, seq_len)  # reconstruction
        return x_hat

    def encode(self, x):
        return self.encoder(x)         # for similarity search
