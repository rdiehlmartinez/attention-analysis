__author__ = 'Richard Diehl Martinez'

'''
Creates a param and experiment object which are used in order to run different
types of analysis on the attention distributions of a given BERT model.
'''

######### General Imports #########

from src.params import Params, read_params
from src.experiment import AttentionExperiment, ClassificationExperiment
from src.dataset import ExperimentDataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

######### Target Model Imports  #########
from models.shallow_nn import ShallowClassifier
from models.full_attentional_nn import FullAttentionalClassifier
from src.utils import logreg_train_for_epoch, logreg_binary_inference_func
from torch.optim import Adam, SGD
from sklearn.linear_model import SGDClassifier
from pytorch_pretrained_bert.modeling import BertForSequenceClassification

######### Model Specific Imports #########
from tasks.bias_classification.lib.shared.data import get_examples, get_sentences_from_dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
import tasks.bias_classification.lib.tagging.model as tagging_model
import tasks.bias_classification.lib.seq2seq.model as seq2seq_model
import tasks.bias_classification.lib.joint.model as complete_model

CUDA = (torch.cuda.device_count() > 0)

############# Experiment specific utils #############

def get_sample_toks(data_path):
    ''' Returns a list of tokens/sentences from the dataset'''
    return get_sentences_from_dataset(data_path)

############# Basic Required Initializations  #############

def intialize_params(experiment_path):
    '''
    Creates a param object that stores the parameters of the given current
    experiment, and creates an Experiment object. The Experiment Object is an
    abstract class that stores the intermediary model from which we can extract
    attention distributions.
    '''
    params = read_params(experiment_path)
    return params

def initialize_classification_experiment(final_task_params):
    '''
    Initializes a classification experiment based on parameters that are
    passed in
    '''
    model_type = final_task_params['model']
    if model_type == 'shallow_nn':
        input_dim = final_task_params['input_dim']
        hidden_dim = final_task_params['hidden_dim']
        output_dim = final_task_params['output_dim']
        model = ShallowClassifier(input_dim, hidden_dim, output_dim)
    elif model_type == 'full_attentional':
        num_attention_dists = final_task_params['num_attention_dists']
        input_dim = final_task_params['input_dim']
        hidden_dim = final_task_params['hidden_dim']
        output_dim = final_task_params['output_dim']
        model = FullAttentionalClassifier(num_attention_dists, input_dim, hidden_dim, output_dim)
    elif model_type == 'bert_basic_uncased_sequence':
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=final_task_params['output_dim'])
    elif model_type == 'log_reg':
        model = SGDClassifier(loss='log')
        train_for_epoch = logreg_train_for_epoch
        inference_func = logreg_binary_inference_func

        classification_experiment = ClassificationExperiment(final_task_params,
                                                             model,
                                                             train_for_epoch_func=train_for_epoch,
                                                             inference_func=inference_func)
        return classification_experiment
    else:
        raise NotImplementedError()

    optimizer = final_task_params['training_params']['optimizer']
    loss_fn = final_task_params['training_params']['loss']

    if optimizer == 'adam':
        optim = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=final_task_params['training_params']['lr'])
    elif optimizer == 'sgd':
        optim = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=final_task_params['training_params']['lr'])
    else:
        raise NotImplementedError()

    if loss_fn == 'bce_with_logits':
        loss_func = F.binary_cross_entropy_with_logits
    elif loss_fn == 'ce_with_logits':
        loss_func = F.cross_entropy
    else:
        raise NotImplementedError()

    return ClassificationExperiment(final_task_params, model, optimizer=optim, loss_fn=loss_func)

def initialize_dataset(intermediary_task_params):
    '''
    Initializes a dataset object which stores useful data for both attention
    extraction and classification.
    '''
    task_specific_params = intermediary_task_params['task_specific_params']
    general_model_params = intermediary_task_params['general_model_params']

    tokenizer = BertTokenizer.from_pretrained(
        task_specific_params['bert_model'],
        cache_dir=task_specific_params['working_dir'] + '/cache')
    tok2id = tokenizer.vocab

    data = get_examples(intermediary_task_params,
                        task_specific_params['target_data'],
                        task_specific_params['target_labels'],
                        tok2id,
                        general_model_params['max_seq_len'])
    return ExperimentDataset(data)

def get_tok2id(intermediary_task_params):
    task_specific_params = intermediary_task_params['task_specific_params']
    tokenizer = BertTokenizer.from_pretrained(
        task_specific_params['bert_model'],
        cache_dir=task_specific_params['working_dir'] + '/cache')
    tok2id = tokenizer.vocab
    return tok2id

def initialize_attention_experiment(intermediary_task_params, verbose=False):
    '''
    Takes in a params object and intializes an Experiment object. To intialize
    an Experiment object, the user has to provide a bert model that has been
    filled out with the learned weights as well as a dataloader which creates
    batches of test data from which attention scores can be derived. The
    user also has to provide a function which given a batch can generate
    token input that can be read in by the extarct_attention_scores method
    of the Experiment class.
    '''
    task_specific_params = intermediary_task_params['task_specific_params']
    general_model_params = intermediary_task_params['general_model_params']

    tokenizer = BertTokenizer.from_pretrained(
        task_specific_params['bert_model'],
        cache_dir=task_specific_params['working_dir'] + '/cache')
    tok2id = tokenizer.vocab
    tok2id['<del>'] = len(tok2id)

    if verbose:
        print("The len of our vocabulary is {}".format(len(tok2id)))
        print("Cuda is set to {}".format('true' if CUDA else 'false'))

    ######################## Specifying Model #####################

    tag_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
                task_specific_params['bert_model'],
                params=intermediary_task_params,
                cls_num_labels=task_specific_params['num_categories'],
                tok_num_labels=task_specific_params['num_tok_labels'],
                cache_dir=task_specific_params['working_dir'] + '/cache',
                tok2id=tok2id)

    if CUDA:
        tag_model = tag_model.eval().cuda()

    debias_model = seq2seq_model.PointerSeq2Seq(
        intermediary_task_params,
        vocab_size=len(tok2id),
        hidden_size=general_model_params['hidden_size'],
        emb_dim=general_model_params['emb_dim'],
        dropout=general_model_params['dropout'],
        tok2id=tok2id)

    if CUDA:
        debias_model = debias_model.eval().cuda()

    checkpoint = torch.load(intermediary_task_params['model_path'])
    joint_model = complete_model.JointModel(intermediary_task_params,
                                            debias_model=debias_model,
                                            tagging_model=tag_model)
    joint_model.load_state_dict(checkpoint)

    if CUDA:
        joint_model.eval()

    def run_inference(batch):
        '''
        Returns the bias classification probabilities for each token and the
        mask that is used for the classification.
        '''
        is_bias_probs, _ = joint_model.run_tagger(
            batch['pre_ids'], batch['masks'],
            rel_ids=batch['rel_ids'],
            pos_ids=batch['pos_ids'],
            categories=batch['categories'])

        return is_bias_probs

    experiment = AttentionExperiment(params=intermediary_task_params['attention'],
                                     bert_model=tag_model,
                                     inference_func=run_inference)

    if verbose:
        print("Succesfully loaded in attention experiment!")
    return experiment
