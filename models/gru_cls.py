__author__ = 'Richard Diehl Martinez'

'''
A recurrent model that reads in attention distribution one attention-distribution
at a time. This is the standard model we use for the majority of our experiments.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(self, params):
        '''
        A basic GRU model that takes in as input attention distributions and
        outputs a prediction for the type of bias. We always assume that the
        input contains some form of bias (either epistemological:0  or framing:1).
        As a result, we frame the classification as a binary classification task.

        Args:
            * params (Param object): See src/params for a definition of this
                class. Effectivelly a wrapper around a json params file.
        '''
        super().__init__()


    def forward(self, X):
