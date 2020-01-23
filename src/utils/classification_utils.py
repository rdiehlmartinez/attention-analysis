__author__ = "Richard Diehl Martinez"

'''
Common inference and training functions for target classification tasks. Also
provides some functionality for analyzing and reporting the resulting classification
statistics.
'''

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm_notebook as tqdm

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

    return_evaluations = label_key != ''

    for step, batch in enumerate(dataloader):
        inputs = batch[input_key]

        if return_evaluations:
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

        if return_evaluations:
            accuracy = np.sum(np.array(batch_predictions) == np.array(labels))/len(labels)
            auc_score = roc_auc_score(labels, predict_probs[:, 1])

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

def run_boostrapping(classification_experiment,
                     dataset,
                     params,
                     input_key='input',
                     label_key='label',
                     threshold=0.42,
                     statistics=["auc", "accuracy"],
                     num_bootstrap_iters=100,
                     **kwargs):
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
                                                                    threshold=threshold,
                                                                    **kwargs)
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
