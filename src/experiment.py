__author__ = 'Richard Diehl Martinez'
'''
Main class that contains abstract methods and functionality required
for any type of experiment. The general framework for an Experiment object is
as follows

Experiment(obj):
    properties:
    * Param(obj): A parameter object which is defined under params.py
    * Bert(NN.Module): A PyTorch model that contains the Bert model which
        has had its learned weights initialized.

    methods:
    * run_inference(input, [mask, optional]): A method that specifies how to
        run inference on the BERT model.
    * extract_attention_scores(): Extract the attention scores from particular
        layers of the BERT model. Which layers attention scores should be
        extracted for is specified bty the params object.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm
import math
import pickle
from src.params import Params
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.shared_utils import CUDA

# Initializing a classification experiment classesmethod
from .utils.classification_utils import logreg_train_for_epoch, logreg_binary_inference_func
from sklearn.linear_model import SGDClassifier
from models.gru_cls import GRUClassifier
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from torch.optim import Adam, SGD

# Initializing an attention experiment classesmethod
import lib.tagging.model as tagging_model
import lib.seq2seq.model as seq2seq_model
import lib.joint.model as complete_model
from  .utils.data_utils import get_tok2id

class Hook():
    '''
    A basic hook class that stores information for each of the intermediary
    modules of a given pytorch model. We use this in order to isolate the
    outputs of individual layers in the BERT model (i.e. attention scores).
    '''
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        self.name = module._get_name()
    def close(self):
        self.hook.remove()

class Experiment(object):
    '''
    Abstract class that holds functionality for any sort of 'Experiment'. An
    Experiment can come in two kids: 1) the first is an experiment involving
    extraction and analysis of attention scores that are derived from the
    intermediary layers of a bert model. 2) the second is an experiment
    involving classification of some type. These two different types of
    'experiments' subclass the more general Experiment class.
    '''
    def __init__(self, params, model):
        self.params = params
        self.model = model

    def set_params(self, params):
        ''' Sets parameters for experiment class '''
        self.params = params

    def set_model(self, model):
        ''' Sets model that'''
        self.model = model

    def set_inference_func(self, func):
        '''
        Given a batch runs inference on the experiment model. The run_inference
        method returns classification probabilities for a particular task,
        but are ignored if only attention layers are required.
        '''
        self._run_inference = func

class ClassificationExperiment(Experiment):
    '''Abstract class that is used for different types of classification tasks'''

    def __init__(self, params, model,
                 loss_fn=None,
                 optimizer=None,
                 train_for_epoch_func=None,
                 inference_func=None):
        super().__init__(params, model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        ClassificationExperiment._train_for_epoch = train_for_epoch_func
        ClassificationExperiment._inference_func = inference_func

        if train_for_epoch_func is None:
            self._train_for_epoch = self.__default_train_for_epoch

        if inference_func is None:
            self._inference_func = self.__default_inference_func

    def reinitialize_weights(self):
        '''Initializing model weights randomly '''
        self.model.apply(ClassificationExperiment.weights_init_uniform_rule)

    @staticmethod
    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if('Linear' in classname):
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def set_train_for_epoch_func(self, func):
        '''
        Users can override the default train for epoch loop.

        Args:
            * func (callabale): must accept a dataloader and kwargs; returns
                a list of dictionary where each evaluation dict should be
                of the form:
                {num_examples: 36, loss: 0.83}
        '''
        self._train_for_epoch = func

    def set_inference_func(self, func):
        '''
        Args:
            * func (callabale): must accept a dataloader and kwargs; returns a
                list or tensor of predictions and a list of dictionary of evaluations

                each evaluation dict should be of the form:
                {num_examples: 36, accuracy: 0.83, ...}
        '''
        self._inference_func = func

    def __default_train_for_epoch(self, dataloader, input_key, label_key, print_every=50, **kwargs):
        ''' Abstract training loop for neural network target task training'''

        assert(self.loss_fn is not None and self.optimizer is not None),\
            "need to implement set_train_func() and set_inference_func()"

        self.model.train()
        losses = []
        for step, batch in enumerate(dataloader):
            if CUDA:
                self.model.cuda()
                batch = {key: val.cuda() for key, val in batch.items()}

            inputs = batch[input_key]
            labels = batch[label_key]

            if 'attention_mask_key' in kwargs:
                masks = batch[kwargs.get("attention_mask_key")]
                predict_logits = self.model(inputs, attention_mask=masks)
            else:
                predict_logits = self.model(inputs)

            if self.params['output_dim'] == 1:
                # in binary case
                labels = torch.reshape(input=labels, shape=predict_logits.shape)
            else:
                # multi class
                #TODO
                raise NotImplementedError("Cannot do classification for multi-class.")

            loss = self.loss_fn(predict_logits, labels.to(dtype=torch.float))

            loss.backward()

            if step % print_every == 0:
                print("Step: {} ; Loss {} ".format(step, loss.item()))

            self.optimizer.step()
            self.model.zero_grad()

            curr_loss = {"num_examples":len(labels),
                         "loss":loss.detach().cpu().item()}

            losses.append(curr_loss)
        return losses

    def __default_inference_func(self, dataloader, input_key, label_key, threshold=0.42, **kwargs):
        ''' inference function for a neural network approach to a target task '''

        assert(self.loss_fn is not None and self.optimizer is not None),\
            "need to implement set_train_func() and set_inference_func()"

        self.model.eval()
        predictions = []
        evaluations = []

        return_evaluations = label_key != ''

        for step, batch in enumerate(dataloader):
            if CUDA:
                self.model.cuda()
                batch = {key: val.cuda() for key, val in batch.items()}

            inputs = batch[input_key]

            if return_evaluations:
                labels = batch[label_key]

            batch_predictions = []
            with torch.no_grad():
                if self.params['output_dim'] == 1:
                    # binary task
                    if 'attention_mask_key' in kwargs:
                        masks = batch[kwargs.get("attention_mask_key")]
                        predict_logits = self.model(inputs, attention_mask=masks)
                    else:
                        predict_logits = self.model(inputs)

                    predict_logits = self.model(inputs).squeeze(1)
                    predict_probs = nn.Sigmoid()(predict_logits).cpu().numpy()

                    for prob in list(predict_probs):
                        if prob > threshold:
                            predictions.append(1)
                            batch_predictions.append(1)
                        else:
                            predictions.append(0)
                            batch_predictions.append(0)

                    if return_evaluations:
                        accuracy = np.sum(np.array(batch_predictions) == np.array(labels))/len(labels)
                        try:
                            auc_score = roc_auc_score(labels, predict_probs)
                        except:
                            print("All labels are of the same type – skipping AUC calculation")
                            continue

                        curr_eval = {"num_examples":len(labels),
                                     "accuracy":accuracy,
                                     "auc":auc_score}
                        evaluations.append(curr_eval)

                else:
                    # multi class --> for bias we are only doing binary classification
                    #TODO
                    raise NotImplementedError()

        return predictions, evaluations

    def train_model(self, train_dataloader, eval_dataloader,
                    input_key="input",
                    label_key="label",
                    threshold=0.42, **kwargs):

        '''Trains self.model using parameters from self.params'''
        num_epochs = self.params['training_params']['num_epochs']
        all_losses = []
        all_evaluations = []

        for epoch in tqdm(range(num_epochs), desc='epochs'):
            keys = {"input_key":input_key,
                    "label_key":label_key,
                    "threshold":threshold}
            kwargs = {**kwargs, **keys}

            losses = self._train_for_epoch(dataloader=train_dataloader, **kwargs)
            predictions, evaluations = self.run_inference(dataloader=eval_dataloader, **kwargs)
            all_losses.append(losses)
            all_evaluations.append(evaluations)

        return all_losses, all_evaluations

    def run_inference(self, dataloader, **kwargs):
        '''Runs inference on self.model and returns model predictions'''
        return self._inference_func(dataloader=dataloader, **kwargs)

    @classmethod
    def init_cls_experiment(cls, final_task_params):
        '''
        Initializes a classification experiment based on parameters that are
        passed in.
        '''

        model_type = final_task_params['model']

        if model_type in ['shallow_nn', 'full_attentional']:
            raise Exception("These model types have been deprecated!")

        elif model_type == 'gru':
            model = GRUClassifier(final_task_params)

        elif model_type == 'transformer':
            raise NotImplementedError()

        elif model_type == 'bert_basic_uncased_sequence':
            # If we are using a BERT model directly for classification
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


class AttentionExperiment(Experiment):
    '''
    Experiment object is an abstract object that contains functionality
    for running analysis on the performance of a given bert model to extract
    attention scores which can then be fed into a final model.
    '''

    def __init__(self, params=None, bert_model=None, inference_func=None):
        super().__init__(params, bert_model)
        self._run_inference = inference_func

    def run_inference(self, dataloader):
        '''
        Runs inference over a dataset. By default runs inference over the
        dataset that is loaded in by the dataloader. Can be overwritten by
        specifying a new dataloader and passing this in for the data optional
        argument.

        Args:
            * dataloader (pytorch dataloader): dataloader for a particular dataset where each
            batch must be processed by self._run_inference

        Returns:
            * results (List): A list of prediction probabilities
        '''

        results = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                prediction_probs = self._run_inference(batch)
                results.append(prediction_probs)
        return results

    @staticmethod
    def transpose_for_scores(x, num_attention_heads, attention_head_size):
        '''
        Helper function to transpose the shape of an attention tensor.
        '''
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def calculate_attention_scores(q_layer, k_layer, num_attention_heads,
                                   attention_head_size, mask=None):
        '''
        Given a set of query, andf matrices, calculates the output
        attention scores.
        '''

        query_layer = AttentionExperiment.transpose_for_scores(q_layer,
                                                      num_attention_heads,
                                                      attention_head_size)
        key_layer = AttentionExperiment.transpose_for_scores(k_layer,
                                                    num_attention_heads,
                                                    attention_head_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if mask is not None:
            mask = mask.to(dtype=torch.float)
            attention_scores = attention_scores + mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        return attention_probs

    def extract_attention_scores(self, dataloader, verbose=False):
        '''
        From the params object, the parameters we need are:
                - num_attention_heads (int): Number of attention heads that
                    should be used in the representation of each attention layers.
                - layers ([int]): A list of indices that specify the attention layers
                    from which we want to extract attention scores.
                - attention_head_size (int): The size of an attention head.
        Returns:
            * attention_scores ({index :{layer index: attention scores tensor}}):
                Returns a dictionary that maps the index of a given sample to a
                dictionary which maps a given layer to the attention scores
                distribution.
        '''
        assert(self._run_inference is not None), "Missing inference function specification"
        assert(self.params is not None), "No params have been specified"
        assert(self.model is not None), "No model specified on which to run inference"


        self._hooks = [Hook(layer[1]) for layer in list(self.model.named_modules())]

        target_layers = self.params['layers']
        num_attention_heads = self.params['num_attention_heads']
        attention_head_size = self.params['attention_head_size']

        attention_scores_list = []

        # TODO: Change requirement of batch size of 1 for attention extraction;
        #       to do this need to think about different tensor dimensionality issue

        with torch.no_grad():
            for step, batch in tqdm(enumerate(dataloader)):
                if CUDA:
                    batch = {key: val.cuda() for key, val in batch.items()}

                mask = batch['masks']

                _ = self._run_inference(batch)
                self_attention_layer = 0

                curr_attention_dict = {}
                for i, hook in enumerate(self._hooks):
                    try:
                        if hook.name == 'BertSelfAttention':
                            if self_attention_layer in target_layers:

                                q_linear = self._hooks[i+1].output
                                k_linear = self._hooks[i+2].output

                                attention_scores = AttentionExperiment.calculate_attention_scores(q_linear,
                                                                                                  k_linear,
                                                                                                  num_attention_heads,
                                                                                                  attention_head_size,
                                                                                                  mask=mask)
                                curr_attention_dict[self_attention_layer] = attention_scores
                            self_attention_layer += 1
                    except AttributeError as e:
                        continue

                attention_scores_list.append(curr_attention_dict)

        return attention_scores_list

    @classmethod
    def initialize_attention_experiment(cls, intermediary_task_params, verbose=False):
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

        tok2id = get_tok2id(intermediary_task_params)
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
