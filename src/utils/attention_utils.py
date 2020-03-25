__author__ = "Richard Diehl Martinez"

"""
Util functions for attention data processing and the extraction of attention
scores from a trained BERT model.
"""

import torch

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
        concat_tensor = torch.cat(tensors, dim=-1)
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

def avg_attention_dist(data):
    '''
    Given a dataset of attentional distributions averages all of the
    attention distributions for each sample.

    Args:
        * data ([{layer index: torch tensor}]): list of dictionaries
            storing the attention distribution for each sample.

    Returns:
        * return ([{index: torch tensor}])
    '''
    num_attention_layers = len(data[0])
    sum_list = sum_attention_dist(data)
    avg_list = [dist/num_attention_layers for dist in sum_list]
    return avg_list

def reduce_attention_dist(data, reducer):
    if reducer == "sum":
        return sum_attention_dist(data)
    elif reducer == "avg":
        return avg_attention_dist(data)
    elif reducer == "concat":
        return concat_attention_dist(data)
    else:
        raise ValueError("Bad parameter: \'reducer\' parameter not in {sum, avg, concat}.")

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
            curr_dict[layer_index] = tensor[:, 0, curr_idx, :]
            # NOTE: attention scores for i^th token are [:,0,i,:] - first dim is thte batch size
        return_list.append(curr_dict)
    return return_list


def window_attention_dist(data, indices, window_size=4):
    '''
    Given a data tensor of reduced attentions, windows the attentions around
    the list of indices passed in.

    ARGS:
        * data: a tensor of attentions, of shape (num_samples, attention_dim)
        * indices: a list of indices around which to window the attention distributions.

    Returns:
        * windowed_data (same type as data)
    '''
    num_samples = data.shape[0]
    additional_padding = torch.zeros((num_samples, window_size))
    expanded_data = torch.cat((additional_padding, data, additional_padding), dim=1)

    indices = torch.tensor(indices) + window_size

    windowed_data = []
    for i, idx in enumerate(indices):
        curr_data = expanded_data[i, idx-window_size:idx+window_size+1]
        windowed_data.append(curr_data)


    windowed_data = torch.stack(windowed_data)
    return windowed_data
