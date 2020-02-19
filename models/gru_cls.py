__author__ = 'Richard Diehl Martinez'

'''
A recurrent model that reads in attention distribution one attention-distribution
at a time. This is the standard model we use for the majority of our experiments.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(self, final_task_params, output_dim=1, bidirectional=True):
        '''
        A basic GRU model that takes in as input attention distributions and
        outputs a prediction for the type of bias. We always assume that the
        input contains some form of bias (either epistemological:0  or framing:1).
        As a result, we frame the classification as a binary classification task.

        Args:
            * params (dictionary of params): These are the params for the classification
                 task passed in from a params object. See src/params for a definition
                 of this class.
            * output_dim (int): Set to 1 since we are doing binary classification.
        '''
        super().__init__()

        # NOTE: we need the input dim to be equal to the max seq len
        self.input_dim = final_task_params['input_dim']
        self.hidden_dim = final_task_params['hidden_dim']
        self.output_dim = output_dim

        self.n_layers = final_task_params['n_layers']
        self.dropout = final_task_params['dropout']
        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          self.n_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)

        self.fc = nn.Linear(self.hidden_dim * (2 if bidirectional else 1), self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # output: (seq_len, batch, num_directions * hidden_size)
        # h_n : (num_layers * num_directions, batch, hidden_size)
        # num_diretions: 2 if bidirectional GRU
        output, hidden = self.gru(x)

        # We concatenate the last hidden layer of the forward and backward model
        hidden_concat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        
        hidden_concat = self.relu(hidden_concat)
        output = self.fc(hidden_concat)
        return output
