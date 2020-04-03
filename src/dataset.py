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
        #all elements should have the same length bc they're stored as tensors
        return len(list(self.data.values())[0])

    def __repr__(self):
        ''' Creates information string about the class '''
        return "Length: {} Keys: ".format(self.__len__()) + str(self.get_key_names())

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
    def init_dataset_without_labels(cls, dataset_params, data_path=''):
        '''
        Initializes a dataset object that does not contain any bias labels. We
        do this when we load in a dataset that will receive labels from the
        weak labeling functions.

        Args:
            * params (a Params object): See params.py.
            * data_path (string): Can override the location of the data to load in.
        '''
        if data_path == '':
            data_path = dataset_params['unlabeled_data']

        tok2id = get_tok2id(dataset_params)
        data = get_examples(dataset_params,
                            data_path,
                            tok2id,
                            dataset_params['max_seq_len'],
                            no_bias_type_labels=True)
        return ExperimentDataset(data)

    @classmethod
    def init_dataset(cls, dataset_params, data_path=''):
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

        if data_path == '':
            data_path = dataset_params['labeled_data']

        tok2id = get_tok2id(dataset_params)
        data = get_examples(dataset_params,
                            data_path,
                            tok2id,
                            dataset_params['max_seq_len'])
        return ExperimentDataset(data)

    @classmethod
    def merge_datasets(cls, dataset1, dataset2):
        '''
        Merges together two datasets, only keeping the keys that are present in
        both dataset1 and dataset2. In effect, this method creates an inner-join
        of the two dataset and returns a new dataset to the user.

        Args:
            dataset1 (ExperimentDataset): First dataset to join
            dataset2 (ExperimentDataset): Second dataset to join

        Returns:
            joined_dataset (ExperimentDataset): A joined version of the datasets
        '''

        shared_keys = set(dataset1.data.keys()) & set(dataset2.data.keys())

        joined_data = {}

        for key in shared_keys:
            data_1 = dataset1.data[key]
            data_2 = dataset2.data[key]

            if isinstance(data_1, np.ndarray):
                data_1 = torch.tensor(data_1)

            if isinstance(data_2, np.ndarray):
                data_2 = torch.tensor(data_2)

            data_1 = data_1.float()
            data_2 = data_2.float()

            joined_data[key] = torch.cat((data_1, data_2), dim=0)

        return ExperimentDataset(joined_data)

    @classmethod
    def split_dataset(cls, dataset1, idx):
        '''
        Splits all of the data in a given dataset at a particular index, and
        returns the new segmented data as a new dataset.

        Args:
            dataset1 (ExperimentDataset): The dataset that we want to split
            idx (int): The index at which to split our data. If negative this
                index will split using data[idx:], otherwise it will split
                using data[:idx].
        '''

        if idx > 0:
            split_data = {key: val[:idx] for key, val in dataset1.data.items()}
        else:
            split_data = {key: val[idx:] for key, val in dataset1.data.items()}
        return ExperimentDataset(split_data)
