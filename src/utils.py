__author__ = 'Richard Diehl Martinez'

'''
Utils file for processing attentional scores and other helper functions.
'''

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import roc_auc_score
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier

################### Data Analysis Utils ###################

def average_data(data):
    '''
    For list of dictionary values, computes the weighted average. Assumes
    that there exists a num_examples key in the dictionary.
    '''
    assert('num_examples' in data[0].keys()), "Data needs to contain a num_examples key"
    average_dict = {key: 0 for key in data[0].keys()}
    total_samples = sum([batch_data['num_examples'] for batch_data in data])
    average_dict['num_examples'] = total_samples
    for batch_data in data:
        batch_num_examples = batch_data['num_examples']
        for key, val in batch_data.items():
            if key == 'num_examples':
                continue
            average_dict[key] += val * batch_num_examples/total_samples
    return average_dict

def get_statistics(data_list, key):
    '''
    Given a list of data statistics returns average, min, max of these.
    '''
    avg_entry = sum([entry[key] for entry in data_list])/len(data_list)
    min_entry = min([entry[key] for entry in data_list])
    max_entry = max([entry[key] for entry in data_list])
    return(min_entry, max_entry, avg_entry)

################### Data Processing Utils ###################

# For bias classification task
def sentence_to_POS_matrix(input_toks, bias_label, indices):
    '''
    Given sentences transforms these into POS tags and then into a dict vectorizer.
    The input tokens are passed in as a dictionary where the key represents a
    particular index in the original dataset and the corresponding value is the
    tokenized sentence.
    '''
    # Typically we want these features to be added in at the start
    # of the dataset
    labels = []
    for i, idx in enumerate(indices): # a list of indices in the dataset

        idx = idx.item()
        assert(idx in input_toks.keys()), \
            "there is an index key in the dataset for which no matching sentence exists"

        sentence_toks = input_toks[idx]
        bias_index = bias_label[i] # the index of the biased word in the sentence
        labels.append({nltk.pos_tag(sentence_toks)[bias_index][1] : 1})
    return DictVectorizer().fit_transform(labels).toarray()

################### Prepare Model Utils ###################

'''
Common inference and training functions for target task models.
'''

def logreg_train_for_epoch(self, dataloader, input_key, label_key, **kwargs):
    accuracies = []
    label_classes = np.unique(dataloader.dataset.data[label_key])
    for step, batch in enumerate(dataloader):
        inputs = batch[input_key]
        labels = batch[label_key]
        self.model = self.model.partial_fit(inputs, labels, classes=label_classes)
        accuracies.append({"num_examples":len(labels),
                           "accuracy":self.model.score(inputs, labels)})
    return accuracies

def logreg_binary_inference_func(self, dataloader, input_key, label_key, threshold=0.42, **kwargs):
    ''' For logistic regression binary-class inference'''
    predictions = []
    evaluations = []
    for step, batch in enumerate(dataloader):
        inputs = batch[input_key]
        labels = batch[label_key]

        predict_probs = self.model.predict_proba(inputs)

        batch_predictions = []
        for prob in list(predict_probs):
            if prob[1] > threshold:
                predictions.append(1)
                batch_predictions.append(1)
            else:
                predictions.append(0)
                batch_predictions.append(0)

        accuracy = np.sum(np.array(batch_predictions) == np.array(labels))/len(labels)
        auc_score = roc_auc_score(labels, predict_probs[:,1])

        curr_eval = {"num_examples":len(labels),
                     "accuracy":accuracy,
                     "auc":auc_score}
        evaluations.append(curr_eval)
    predictions = np.stack(predictions, axis=0)
    return predictions, evaluations

def logreg_multi_inference_func(self, dataloader, input_key, label_key, **kwargs):
    ''' For logistic regression multi-class inference'''
    predictions = []
    evaluations = []
    for step, batch in enumerate(dataloader):
        inputs = batch[input_key]
        labels = batch[label_key]
        predictions.append(self.model.predict_proba(inputs, labels))
        accuracy = self.model.score(inputs, labels)
        curr_eval = {"num_examples":len(labels),
                     "accuracy":accuracy}
        evaluations.append(curr_eval)
    predictions = np.stack(predictions, axis=0)
    return predictions, evaluations


################### Classification Utils ###################

def run_boostrapping(classification_experiment,
                     dataset,
                     params,
                     input_key='input',
                     label_key='label',
                     threshold=0.42,
                     statistics=["auc", "accuracy"],
                     num_bootstrap_iters=100):
    '''
    Randomly shuffles dataset and reports confidence interval statistics
    via using bootstrapping methods.
    '''
    stats_list = {statistic: [] for statistic in statistics}
    for _ in tqdm(range(num_bootstrap_iters), desc='Bootstrapping'):
        dataset.shuffle_data()
        data_split = params.final_task['data_split']
        batch_size = params.final_task['training_params']['batch_size']
        loaders = dataset.split_train_eval_test(**data_split, batch_size=batch_size)
        train_dataloader, eval_dataloader, _ = loaders
        if params.final_task["model"] == "log_reg":
            classification_experiment.model = SGDClassifier(loss='log')
        else:
            classification_experiment.reinitialize_weights()

        _, evaluations = classification_experiment.train_model(train_dataloader,
                                                                    eval_dataloader,
                                                                    input_key=input_key,
                                                                    label_key=label_key,
                                                                    threshold=threshold)
        avg_evaluations = [average_data(epoch_evaluations) for epoch_evaluations in evaluations]
        for statistic in statistics:
            _, max_statistic, _ = get_statistics(avg_evaluations, statistic)
            stats_list[statistic].append(max_statistic)

    return_stats = {}
    for stat, values in stats_list.items():
        lower_ci_bound = np.percentile(values, 2.5)
        upper_ci_bound = np.percentile(values, 97.5)
        avg_val = np.mean(values)
        return_stats[stat] = [(lower_ci_bound, upper_ci_bound), avg_val]
    return return_stats


################### Attention Extraction Utils ###################

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
