__author__ = 'Richard Diehl Martinez'
'''
Basic shallow classifier that performs classification for some task
given an attention distribution.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        '''
        output dim is set to 1 for binary cross entropy loss.
        '''
        super().__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        hidden = F.relu(self.f1(X))
        output = self.f2(hidden)
        return output
    