__author__ = 'Richard Diehl Martinez'

'''
Provides a number of useful utility functions and classes for data processing,
model intialization and model training procedures.
'''

import json

################# Parameter Processing #################

class Params(object):
    '''Establishes wrapper class around parameter json file.'''
    def __init__(self, data):
        self.__dict__ = json.load(data)

    def __repr__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=3)

def read_params(path):
    '''
    Reads in json parameter file and created a Params Object.
    '''
    assert('.json' in path), "file passed into read_params() is not in json format"
    with open(path) as json_file:
        return Params(json_file)

def intialize_params(experiment_path):
    '''
    Creates a param object that stores the parameters of the given current
    experiment, and creates an Experiment object. The Experiment Object is an
    abstract class that stores the intermediary model from which we can extract
    attention distributions.
    '''
    params = read_params(experiment_path)
    return params
