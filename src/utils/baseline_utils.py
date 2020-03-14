__author__ = 'Richard Diehl Martinez'

'''
Util file that stores functions used in generating some of the baselines. Our
baselines are generaally of the format: (input sentence) -> (prediction). We have no
access to what the actual biased word is, and instead convert the entire
sentence into some embedding before feeding that as input to a classification
layer.

We implement one difficult baseline. That uses the BERT pretrained bias detection
module to first predict what the biased word is and then transforms that word
into a feature vector using the Marta Recasens' feature engineered features.
'''

import torch
from .data_utils import get_words_and_indices, get_id2tok, get_tok2id
from src.utils.shared_utils import CUDA

def get_bias_predictions(dataset, intermediary_task_params, dataset_params):
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

    checkpoint = torch.load(intermediary_task_params['model_path'])
    joint_model.load_state_dict(checkpoint)
    joint_model.eval()

    def run_inference(entry):
        '''
        Returns the bias classification probabilities for each token and the
        mask that is used for the classification.
        '''
        batch = {key: val.unsqueeze(0).cuda() for key, val in entry.items()}
        cls_logits, tok_logits = joint_model.run_tagger(
             batch['pre_ids'], batch['masks'],
             rel_ids=batch['rel_ids'],
             pos_ids=batch['pos_ids'],
             categories=batch['categories'])
        return(cls_logits, tok_logits)

    predictions_label_ids = []
    for entry in dataset:
        cls_logits, tok_logits = run_inference(entry)
        arg_max_cls = cls_logits.squeeze().argmax(0).cpu()
        predicted_pre_tok_label_ids = entry['pre_tok_label_ids']

        correct_bias_idx = predicted_pre_tok_label_ids.to(dtype=torch.int).flatten().tolist().index(1)
        predicted_pre_tok_label_ids[correct_bias_idx] = 0
        predicted_pre_tok_label_ids[arg_max_cls] = 1
        predictions_label_ids.append(predicted_pre_tok_label_ids)
    return torch.stack(predictions_label_ids)

def get_bow_matrix(dataset_params, dataset):
    '''
    Iterates over a dataset and all of the words in each sentence creates a
    bag of words representation of the input. We return a large matrix which
    can then be added back into the dataset which is in turn fed into a
    classification model.

    Args:
        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * bow_matrix (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset.
    '''

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()

    sentences = []

    id2tok = get_id2tok(dataset_params)

    for entry in dataset:
        pre_ids = entry["pre_ids"].tolist() # the ids of the tokens prior to debias edits
        pad_idx = pre_ids.index(0) # specify that 0 is padding id
        toks = [id2tok[x] for x in pre_ids[:pad_idx]]
        words, tok_to_word_idx = get_words_and_indices(toks)

        sentences.append(' '.join(words))

    # dataset_size, num_vocab
    pos_matrix = vectorizer.fit_transform(sentences).toarray()
    pos_tensor = torch.tensor(pos_matrix, dtype=torch.float32)

    return pos_tensor
