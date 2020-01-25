__author__ = 'Richard Diehl Martinez'

'''
Util file that stores useful functions and variables used accross all experiments.
---
Most specific util functions should be defined in the other util files.
'''

import torch

CUDA = (torch.cuda.device_count() > 0)
