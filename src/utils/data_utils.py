__author__ = "Richard Diehl Martinez"

'''
Util functions to help with data processing and data extraction.
'''

import nltk
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

def get_sample_toks(data_path):
    ''' Returns a list of tokens/sentences from the dataset'''
    return get_sentences_from_dataset(data_path)

def get_tok2id(intermediary_task_params):
    task_specific_params = intermediary_task_params['task_specific_params']
    tokenizer = BertTokenizer.from_pretrained(
        task_specific_params['bert_model'],
        cache_dir=task_specific_params['working_dir'] + '/cache')
    tok2id = tokenizer.vocab
    return tok2id

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

# For bias classification task
def sentence_to_POS_matrix(input_toks,
                           bias_label,
                           indices,
                           return_pos_list=False,
                           valid_pos_label=None):
    '''
    Given sentences transforms these into POS tags and then into a dict vectorizer.
    The input tokens are passed in as a dictionary where the key represents a
    particular index in the original dataset and the corresponding value is the
    tokenized sentence.
    '''
    # Typically we want these features to be added in at the start
    # of the dataset
    labels = []
    pos_tags = []
    skip_indices = []
    for i, idx in tqdm(enumerate(indices)): # a list of indices in the dataset

        idx = idx.item()

        if idx not in input_toks.keys():
            print(idx)

        assert(idx in input_toks.keys()), \
            "there is an index key in the dataset for which no matching sentence exists"

        sentence_toks = input_toks[idx]
        bias_index = bias_label[i] # the index of the biased word in the sentence
        pos_label = nltk.pos_tag(sentence_toks)[bias_index][1]

        if valid_pos_label is not None and pos_label not in valid_pos_label:
            skip_indices.append(i)
            continue

        pos_tags.append(pos_label)
        labels.append({pos_label : 1})

    if valid_pos_label is not None:
        return (DictVectorizer().fit_transform(labels).toarray(), skip_indices)

    # In case the user needs easy access to the set of pos tag labels
    if return_pos_list:
        return DictVectorizer().fit_transform(labels).toarray(), set(pos_tags)
    else:
        return DictVectorizer().fit_transform(labels).toarray()
