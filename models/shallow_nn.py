__author__ = 'Richard Diehl Martinez'
'''
Basic shallow classifier that performs classification for some task
given some low-dimensional input. Typically, we will use this classifier
for baseline and weak-labeling classifiers.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowClassifier(nn.Module):
    def __init__(self, final_task_params, output_dim=1):
        '''
        A basic feed forward model that takes in a low-dimensional input and
        outputs a prediction for the type of bias. We always assume that the
        input contains some form of bias (either epistemological:0 or framing:1).
        As a result, we frame the classification as a binary classification task.

        Args:
            * params (dictionary of params): These are the params for the classification
                 task passed in from a params object. See src/params for a definition
                 of this class.
            * output_dim (int): Set to 1 since we are doing binary classification.
        '''
        super().__init__()

        self.input_dim = final_task_params['input_dim']
        self.hidden_dim = final_task_params['hidden_dim']
        self.output_dim = output_dim

        self.f1 = nn.Linear(self.input_dim , self.hidden_dim)
        self.f2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, X, **kwargs):
        hidden = self.relu(self.f1(X))
        output = self.f2(hidden)
        return output
