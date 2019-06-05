__author__ = 'Richard Diehl Martinez'
'''
Applies an affine combination over a set of BERT attention distributions.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullAttentionalClassifier(nn.Module):
    def __init__(self, num_attention_dists, input_dim, hidden_dim, output_dim=1):
        '''
        output dim is set to 1 for binary cross entropy loss.
        '''
        super().__init__()
        self.weighted_combination = nn.Linear(num_attention_dists, 1)
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.f3 = nn.Linear(int(hidden_dim/2), output_dim)

    def forward(self, X):
        transpose_input = torch.transpose(X, -2, -1)
        weighted_combination = self.weighted_combination(transpose_input)
        weighted_combination = torch.transpose(weighted_combination, -2, -1)
        hidden = F.relu(self.f1(weighted_combination))
        hidden = F.relu(self.f2(hidden))
        output = self.f3(hidden)
        return output
