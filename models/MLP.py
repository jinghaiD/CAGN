import numpy as np
import torch
from torch.autograd.grad_mode import no_grad
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F

'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(64,64), activation='relu', dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.outlayer = nn.Linear(output_dim, output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            
        return x