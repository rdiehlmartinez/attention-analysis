__author__ = 'Richard Diehl Martinez'

'''
Util file that stores useful functions and variables used for the weak
labeling functions.
'''

import numpy as np
import torch

def get_marta_featurizer(dataset_params):
    '''
    Returns the featurizer implement by Pryzant et al. for extracting the lingutistic
    features defined by Recasens. et al. This featurizer is stored under
    lib/tagging/features.py.

    Args:

        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
    '''

    from lib.tagging.features import Featurizer
    from .data_utils import get_tok2id

    assert("lexicon_dir" in dataset_params.keys()), \
        "Lexicon directory path must be specified in the dataset parameters"

    tok2id = get_tok2id(dataset_params)
    return Featurizer(tok2id, params=dataset_params)

def extract_marta_features(dataset, featurizer):
    '''
    Extract Marta's set of linguistic features from a particular dataset.

    Args:
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids
        * featurizer (Featurizer obj): defined in lib/tagging/features

    Returns:
        * tensor_features (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    full_features = []
    for entry in dataset:
        features = featurizer.features(entry["pre_ids"].tolist(),
                                       entry["rel_ids"].tolist(),
                                       entry["pos_ids"].tolist())
        # Figuring out what the index is of the first biased word
        bias_idx = entry['pre_tok_label_ids'].to(dtype=torch.int).flatten().tolist().index(1)
        full_features.append(features[bias_idx, :])
    # num_entries, dim
    tensor_features = torch.tensor(np.stack(full_features), dtype=torch.float32)
    return tensor_features

def get_glove_features(dataset):
    '''
    Iterates over a dataset and for the predicted biased word generates the GloVe
    embedding of that word.

    Args:
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * tensor_features (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    raise NotImplementedError()
