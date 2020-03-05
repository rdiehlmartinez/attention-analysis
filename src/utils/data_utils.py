__author__ = "Richard Diehl Martinez"

'''
Util functions to help with data processing and data extraction.
'''

import nltk
import torch
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.feature_extraction import DictVectorizer
from pytorch_pretrained_bert.tokenization import BertTokenizer
from lib.shared.data import get_sentences_from_dataset

str_to_dtype = {
    "long":torch.long,
    "uint8":torch.uint8,
    "float":torch.float,
    "int16":torch.int16,
    "int8":torch.int8
}

def get_bias_indices(labels):
    '''
    Returns the index of of the first occurrence of a '1' in a tensor. Useful
    for finding the first occurrence of a biased word.
    '''
    bias_indices = [label.flatten().tolist().index(1) for label in labels]
    return bias_indices

def get_sample_toks(data_path):
    ''' Returns a list of tokens/sentences from the dataset'''

    # NOTE: this might be deprecated
    return get_sentences_from_dataset(data_path)

def get_tok2id(dataset_params):
    ''' Creates a dict mapping of tokens to their corresponding BERT ids '''
    tokenizer = BertTokenizer.from_pretrained(
        dataset_params['bert_model'],
        cache_dir=dataset_params['working_dir'] + '/cache')
    tok2id = tokenizer.vocab
    return tok2id

def get_id2tok(dataset_params):
    ''' Creates a dict mapping of BERT itds to their corresponding tokens '''
    tok2id = get_tok2id(dataset_params)
    id2tok = {x: tok for tok, x in tok2id.items()}
    return id2tok

def glove2dict(src_filename):
    """GloVe Reader.
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors.
    """
    data = {}
    with open(src_filename) as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def get_words_and_indices(toks):
    '''
    Helper function that for a token sequence builds a list of
    [words, [tok indices the word came from]].

    Args:
        * toks ([Int]): A list containing the ids of the tokens in a particular entry.
    Returns
        * words ([String]): A list of word strings (joined together tokens).
        * tok_to_word ({tok index a word came from: word indices}): Maps
            token indices to word indices.
    '''
    words = []
    tok_to_word = {}
    word_counter = 0
    for idx, tok in enumerate(toks):
        if tok.startswith('##'):
            word_counter -= 1
            words[-1] += tok.replace('##', '')
        else:
            words.append(tok)
        tok_to_word[idx] = word_counter
        word_counter += 1

    return (words, tok_to_word)
