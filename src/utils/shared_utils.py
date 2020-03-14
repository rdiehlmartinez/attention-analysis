__author__ = 'Richard Diehl Martinez'

'''
Util file that stores useful functions and variables used accross all experiments.
---
Most specific util functions should be defined in the other util files.
'''

import torch

CUDA = (torch.cuda.device_count() > 0)
from .data_utils import get_tok2id

def load_bias_detection_module(intermediary_task_params, dataset_params):
    '''
    Returns a pretrained tagging module - that is a BERT model which has been
    pretrained to detect bias in a sentence.

    Args:
        * intermediary_task_params (dict): Dictionary of parameters for
             the intermediary model from which we will extract the attention
             scores.
        * dataset_params (dict): Dictionary of parameters for the dataset
            that will be used to extract attention scores from.
    Return:
        * joint_model : see lib.joint.model for model details
    '''

    import lib.tagging.model as tagging_model
    import lib.seq2seq.model as seq2seq_model
    import lib.joint.model as complete_model

    general_model_params = intermediary_task_params['general_model_params']

    tok2id = get_tok2id(dataset_params)
    tok2id['<del>'] = len(tok2id)

    tag_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
                general_model_params['bert_model'],
                params=general_model_params,
                cls_num_labels=dataset_params['num_categories'],
                tok_num_labels=dataset_params['num_tok_labels'],
                cache_dir=general_model_params['working_dir'] + '/cache',
                tok2id=tok2id)


    if CUDA:
        tag_model = tag_model.eval().cuda()

    debias_model = seq2seq_model.PointerSeq2Seq(
        params=general_model_params,
        vocab_size=len(tok2id),
        hidden_size=general_model_params['hidden_size'],
        emb_dim=general_model_params['emb_dim'],
        dropout=general_model_params['dropout'],
        tok2id=tok2id)

    if CUDA:
        debias_model = debias_model.eval().cuda()

    joint_model = complete_model.JointModel(params=general_model_params,
                                            debias_model=debias_model,
                                            tagging_model=tag_model)

    return joint_model

def run_bias_detection_inference(joint_model, batch):
    '''
    Returns the bias classification probabilities for each token and the
    mask that is used for the classification.
    Args:
        * joint_model: pretrained joint_model specified in lib.tagging_model.model
        * batch: A batch returned by an experiment datast with all of the
            associated entries.
    '''
    if CUDA:
        batch = {key: val.cuda() for key, val in batch.items()}

    cls_logits, tok_logits = joint_model.run_tagger(
         batch['pre_ids'], batch['masks'],
         rel_ids=batch['rel_ids'],
         pos_ids=batch['pos_ids'],
         categories=batch['categories'])
    return(cls_logits, tok_logits)

def get_bias_predictions(dataset, intermediary_task_params, dataset_params, **kwargs):
    '''
    Returns the predicted biased word from a dataset using the tagger which
    has been pre-trained on bias detection.

    Args:
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids
        * intermediary_task_params (dict): Dictionary of parameters for
             the intermediary model from which we will extract the attention
             scores.
        * dataset_params (dict): Dictionary of parameters for the dataset
            that will be used to extract attention scores from.
    '''

    joint_model = load_bias_detection_module(intermediary_task_params, dataset_params)

    checkpoint = torch.load(intermediary_task_params['model_path'])
    joint_model.load_state_dict(checkpoint)
    joint_model.eval()

    predictions_label_ids = []

    for entry in dataset.return_dataloader(**kwargs):
        cls_logits, tok_logits = run_bias_detection_inference(joint_model, entry)
        arg_max_cls = cls_logits.squeeze().argmax(1).cpu()

        #return
        predicted_pre_tok_label_ids = entry['pre_tok_label_ids']
        correct_bias_idx = predicted_pre_tok_label_ids.to(dtype=torch.int).flatten().tolist().index(1)
        predicted_pre_tok_label_ids[:, correct_bias_idx] = 0
        predicted_pre_tok_label_ids[:, arg_max_cls] = 1
        print(predicted_pre_tok_label_ids.shape)
        predictions_label_ids.append(predicted_pre_tok_label_ids)

    return torch.cat(predictions_label_ids, dim=0)
