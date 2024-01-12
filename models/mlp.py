# Libraries
import torch
import torch.nn as nn
from torch.nn import Linear


class MLP(nn.Module):
    def __init__(self, hid_channels, num_inputs, num_outputs, num_layers=2, dropout_rate=0):
        super(MLP, self).__init__()
        torch.manual_seed(42)
        self.hid_channels = hid_channels
        self.dropout_rate = dropout_rate
        
        layers = [Linear(num_inputs, hid_channels),
                  nn.ReLU(),
                  nn.Dropout(self.dropout_rate)]
        
        for l in range(num_layers-1):
            layers += [Linear(hid_channels, hid_channels),
                       nn.ReLU(),
                       nn.Dropout(self.dropout_rate)]
            
        layers += [Linear(hid_channels, num_outputs)]
        
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.main(x)
        
        return x