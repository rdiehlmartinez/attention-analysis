__author__ = 'Richard Diehl Martinez'
'''
Applies an affine combination over a set of BERT attention distributions.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullAttentionalClassifier(nn.Module):
    def __init__(self, num_attention_dists, attention_dim, hidden_dim, output_dim=1):
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
