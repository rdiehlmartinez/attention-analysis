__author__ = "Richard Diehl Martinez"

"""
Util functions for attention data processing and the extraction of attention
scores from a trained BERT model.
"""

import torch


str_to_dtype = {
    "long":torch.long,
    "uint8":torch.uint8,
    "float":torch.float,
    "int16":torch.int16,
    "int8":torch.int8
}

def attention_score_window(data, indices, window_size=5):
    ''' Default method for processing attention_scores. '''

    inputs = []
    labels = []
    skipped_indicies = []
    for i, entry in enumerate(data[0]):
        entry_len = entry.shape[0]
        curr_idx = indices[i]
        if(curr_idx - window_size < 0 or curr_idx + window_size + 1> entry_len-1):
            skipped_indicies.append(i)
            continue

        curr_input = entry[curr_idx-window_size:curr_idx+window_size+1]
        labels.append(data[-1][i])
        inputs.append(curr_input)

    return(inputs, labels, skipped_indicies)

def window_attention_dist(data, indices, window_size):
    '''
    Given a list of data which can either be dict of tensors or tensors,
    creates a window around the index that is passed in.

    ARGS:
        * data: either a list of tensors or a list of dictionaries
            storing the attention distribution for each sample.

    Returns:
        * windowed_data (same type as data)
        * removed_samples ([int]): a list of indices of the removed sample indices
    '''
    assert(len(data) > 0), "data is empty"

    removed_samples = []
    windowed_data = []

    if isinstance(data[0], torch.Tensor):
        for i, entry in enumerate(data):
            entry_len = entry.shape[0]
            curr_idx = indices[i]
            if(curr_idx - window_size < 0 or curr_idx + window_size + 1 > entry_len-1):
                removed_samples.append(i)
                continue

            if len(list(entry.shape)) == 2:
                curr_input = entry[:, curr_idx-window_size:curr_idx+window_size+1]
            elif len(list(entry.shape)) == 1 :
                curr_input = entry[curr_idx-window_size:curr_idx+window_size+1]
            else:
                raise Exception("what's good with this 3d attention dist - what you doing?")
            windowed_data.append(curr_input)

    elif isinstance(data[0], dict):
        for i, sample_dict in enumerate(data):
            entry_len = list(sample_dict.values())[0].shape[0]
            curr_dict = {}
            curr_idx = indices[i]
            if(curr_idx - window_size < 0 or curr_idx + window_size + 1> entry_len-1):
                removed_samples.append(i)
                continue

            for (layer_index, tensor) in sample_dict.items():
                if len(list(tensor.shape)) == 2:
                    curr_input = tensor[:, curr_idx-window_size:curr_idx+window_size+1]
                elif len(list(tensor.shape)) == 1:
                    curr_input = tensor[curr_idx-window_size:curr_idx+window_size+1]
                else:
                    raise Exception("what's good with this 3d attention dist - what you doing?")
                curr_dict[layer_index] = curr_input
            windowed_data.append(curr_dict)

    else:
        raise Exception("Invalid datatype passed in")

    return(windowed_data, removed_samples)

def concat_attention_dist(data):
    '''
    Given a dataset of attentional distributions concats together the
    attention distributions for each sample. This method is suggested as the
    best for feature extracting by Devlin et al., 2018. Need to ensure that the
    tensors passed in are of correct dimensions (i.e. are windows of some kind)

    Args:
        * data ([{layer index: torch tensor}]): list of dictionaries
            storing the attention distribution for each sample.

    Returns:
        * return ({index: torch tensor})
    '''
    return_list = []
    for sample_dict in data:
        _, tensors = zip(*sample_dict.items())
        concat_tensor = torch.cat(tensors)
        return_list.append(concat_tensor)
    return return_list


def sum_attention_dist(data):
    '''
    Given a dataset of attentional distributions sums together all of the
    attention distributions for each sample.

    Args:
        * data ([{layer index: torch tensor}]): list of dictionaries
            storing the attention distribution for each sample.

    Returns:
        * return ([{index: torch tensor}])
    '''
    return_list = []
    for sample_dict in data:
        _, tensors = zip(*sample_dict.items())
        sum_tensor = torch.stack(tensors, dim=0).sum(dim=0)
        return_list.append(sum_tensor)
    return return_list

def return_idx_attention_dist(data, indices):
    '''
    Given a set of indices for each example returns the attention distribution
    from a data dictionary for that particular index. For instance, in the case
    of bias classification we want to return the attention distribution that is
    associated only with the biased word.

    Args:
        * data ([{layer index: torch tensor}]): list of dictionaries
            storing the attention distribution for each sample.

    Returns:
        * return (same as data)
    '''

    return_list = []
    for i, sample_dict in enumerate(data):
        curr_idx = indices[i]
        curr_dict = {}
        for (layer_index, tensor) in sample_dict.items():
            curr_dict[layer_index] = tensor[0, 0, curr_idx, :]
            # NOTE: attention scores for i^th token are [0,0,i,:]
        return_list.append(curr_dict)
    return return_list
