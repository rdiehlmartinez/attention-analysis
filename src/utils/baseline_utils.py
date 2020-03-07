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
from .data_utils import get_words_and_indices, get_id2tok

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
