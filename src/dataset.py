__author__ = 'Richard Diehl Martinez'
'''
Specifies an abstract dataset class that stores data which can be used
to extract attention scores from and train a final target task for. Once
intialized with a set of data, this dataset class cannot be used to add
additional data to the dataset. Instead, if a model is first trained on a
small dataset and then from that a labeling function is derived, this will
generate labels for a new dataset class that has to be initialized with a
greater amount of data.
'''

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, Dataset
from lib.shared.data import get_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer
from src.utils.data_utils import get_tok2id, str_to_dtype

class ExperimentDataset(Dataset):
    ''' Datasets used for experiments.'''

    def __init__(self, data):
        '''
        Data is stored as a dictionary; with different values depending on the
        type of data that is required. User must ensure that the data has
        correct proporitions. All entries in data must be stored in some
        numpy matrix or pytorch tensor (no python lists).

        Recommended that users pass in a list of indices that match up with the
        indices of the entries in the original dataset.
        '''
        self.data = data

        for key, entry in self.data.items():
            if not isinstance(entry, torch.Tensor) and not isinstance(entry, np.ndarray):
                raise Exception("{} is not torch tensor or numpy array".format(key))

        if 'index' not in self.data.keys():
            print("NOTICE: You may want to specify an index or id value for each sample")

    def __getitem__(self, idx):
        ''' Required by torch.Dataset '''
        sample = {}
        for key, value in self.data.items():
            sample[key] = value[idx]
        return sample

    def __len__(self):
        ''' Required by torch.Dataset '''
        #all elements should have the same length
        return len(list(self.data.values())[0])

    def remove_indices(self, indices):
        '''
        Removes the indices that are passed in as a list, from all of the
        entries in the dataset.
        '''
        all_indices = np.arange(len(self))
        keep_indices = np.delete(all_indices, indices)
        for key in self.data.keys():
            self.data[key] = self.data[key][keep_indices]

    def get_val(self, key):
        ''' dictionary look up of a particular value'''
        return self.data[key]

    def get_key_names(self):
        ''' Returns the current keys that stored in the dataset '''
        return self.data.keys()

    def return_dataloader(self, data=None, batch_size=1, **kwargs):
        '''
        Creates a dataloader from the data passed in. Need to specify a set
        of keys to select the data from the dataset that should be used in
        the dataloader.
        '''

        if data is None:
            dataset = self
        else:
            dataset = ExperimentDataset(data)

        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=batch_size,
            **kwargs)

        return dataloader

    def add_data(self, data, key_name):
        ''' Adds column of data to datset'''
        self.data[key_name] = data

    def shuffle_data(self):
        '''Shuffles the data randomly'''
        p = np.random.permutation(len(self))
        for key in self.data.keys():
            self.data[key] = self.data[key][p]

    def split_train_eval_test(self, data=None, train_split=0.8, eval_split=0.1, test_split=0.1, batch_size=1):
        '''
        Creates a split of the dataset into train, eval and test; and returns
        three dataloaders with these corresponding split_train_eval_test.
        '''

        if data is None:
            data = self.data



        train_data = {key: val[:int(len(self) * train_split)] for key, val in data.items()}
        eval_data = {key: val[int(len(self) * train_split):int(len(self) * (train_split+eval_split))] \
                                                    for key, val in data.items()}
        test_data = {key: val[int(len(self) * (train_split+eval_split)):] \
                                                    for key, val in data.items()}
        assert(len(set(map(int, train_data["index"])) & set(map(int, eval_data["index"]))) == 0)
        assert(len(set(map(int, train_data["index"])) & set(map(int, test_data["index"]))) == 0)

        train_dataloader = self.return_dataloader(data=train_data, batch_size=batch_size)
        eval_dataloader = self.return_dataloader(data=eval_data, batch_size=batch_size)
        test_dataloader = self.return_dataloader(data=test_data, batch_size=batch_size)

        return (train_dataloader, eval_dataloader, test_dataloader)

    @classmethod
    def init_dataset(cls, params, data_path=''):
        '''
        Initializes a dataset object which stores data for both attention
        extraction and classification. The bulk of the data parsing is done
        by the get_examples() function.

        Note:
        The general idea is that we always load in a dataset that has the tokens
        as well as the labels - in case we are creating weak labels we can do
        this too in the same way. The work of distinguishing if we have the
        correct labels or not should be done in the get_examples function.

        Args:
            * params (a Params object): See params.py.
            * data_path (string): Can override the location of the data to load in.
        '''

        data_path = params.final_task['labeled_data']
        general_model_params = params.intermediary_task['general_model_params']

        tok2id = get_tok2id(params.intermediary_task)
        data = get_examples(params.intermediary_task,
                            data_path,
                            tok2id,
                            general_model_params['max_seq_len'])
        return ExperimentDataset(data)
