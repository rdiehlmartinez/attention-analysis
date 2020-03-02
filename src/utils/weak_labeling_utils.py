__author__ = 'Richard Diehl Martinez'

'''
Util file that stores useful functions and variables used for the weak
labeling functions.
'''

import numpy as np
import torch

def get_marta_featurizer(dataset_params):
    '''
    Returns the featurizer implement by Pryzant et al. for extracting the lingutistic
    features defined by Recasens. et al. This featurizer is stored under
    lib/tagging/features.py.

    Args:

        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
    '''

    from lib.tagging.features import Featurizer
    from .data_utils import get_tok2id

    assert("lexicon_dir" in dataset_params.keys()), \
        "Lexicon directory path must be specified in the dataset parameters"

    tok2id = get_tok2id(dataset_params)
    return Featurizer(tok2id, params=dataset_params)

def extract_marta_features(dataset, featurizer):
    '''
    Extract Marta's set of linguistic features from a particular dataset.

    Args:
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids
        * featurizer (Featurizer obj): defined in lib/tagging/features

    Returns:
        * tensor_features (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    full_features = []
    for entry in dataset:
        features = featurizer.features(entry["pre_ids"].tolist(),
                                       entry["rel_ids"].tolist(),
                                       entry["pos_ids"].tolist())
        # Figuring out what the index is of the first biased word
        bias_idx = entry['pre_tok_label_ids'].to(dtype=torch.int).flatten().tolist().index(1)
        full_features.append(features[bias_idx, :])
    # num_entries, dim
    tensor_features = torch.tensor(np.stack(full_features), dtype=torch.float32)
    return tensor_features

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

def load_glove_dict(dataset_params):
    ''' Given a path to a GloVe pretrained vectors file reads these into a dictionary.'''
    glove_dict = {}
    with open(dataset_params['glove_data'], 'r+') as glove_file:
        for line in glove_file:
            line = line.split()
            glove_dict[line[0]] = torch.tensor(np.array(line[1:], dtype=float), dtype=torch.float)
    return glove_dict

def get_glove_features(dataset_params, dataset):
    '''
    Iterates over a dataset and for the predicted biased word generates the GloVe
    embedding of that word.

    Args:
        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * tensor_features (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    from .data_utils import get_id2tok

    embeddings = []
    id2tok = get_id2tok(dataset_params)

    # loading in the GloVe dictionary
    glove_dict = load_glove_dict(dataset_params)

    for entry in dataset:
        pre_ids = entry["pre_ids"].tolist() # the ids of the tokens prior to debias edits
        pad_idx = pre_ids.index(0) # specify that 0 is padding id
        toks = [id2tok[x] for x in pre_ids[:pad_idx]]
        words, tok_to_word_idx = get_words_and_indices(toks)

        # Figuring out what the index is of the first biased word
        bias_token_idx = entry["pre_tok_label_ids"].to(dtype=torch.int).flatten().tolist().index(1)
        bias_word = words[tok_to_word_idx[bias_token_idx]]

        # looking up bias word in GloVe
        try:
            embedding = glove_dict[bias_word]
        except KeyError:
            # if the word is not in the GloVe dictionary replace with all 0s
            embedding = torch.zeros(dataset_params['glove_embedding_dim'], dtype=torch.float)

        embeddings.append(embedding)

    # num_entries, dim
    tensor_embeddings = torch.stack(embeddings)
    return tensor_embeddings


def get_pos_features(dataset_params, dataset):
    '''
    Iterates over a dataset and for the predicted biased word generates the pos
    tag for that word.

    Args:
        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * tensor_features (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    from .data_utils import get_id2tok
    import nltk
    from sklearn.feature_extraction import DictVectorizer

    pos_tags = []
    id2tok = get_id2tok(dataset_params)

    for entry in dataset:
        pre_ids = entry["pre_ids"].tolist() # the ids of the tokens prior to debias edits
        pad_idx = pre_ids.index(0) # specify that 0 is padding id
        toks = [id2tok[x] for x in pre_ids[:pad_idx]]
        words, tok_to_word_idx = get_words_and_indices(toks)

        # Figuring out what the index is of the first biased word
        bias_token_idx = entry["pre_tok_label_ids"].to(dtype=torch.int).flatten().tolist().index(1)
        bias_word_idx = tok_to_word_idx[bias_token_idx]

        # looking up tag of the biased word and creating the POS matrix

        pos_tag = nltk.pos_tag(words)[bias_word_idx][1]
        pos_tags.append({pos_tag: 1})


    pos_matrix = DictVectorizer().fit_transform(pos_tags).toarray()
    # num_entries, num_pos
    pos_matrix = torch.tensor(pos_matrix)
    
    return pos_matrix


def get_bert_features(dataset_params, dataset):
    '''
    Iterates over a dataset and for the predicted biased word generates the
    contextualized BERT embedding for that word.

    Args:
        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * tensor_features (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    from pytorch_pretrained_bert import BertModel

    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []

    for entry in dataset:
        # Figuring out what the index is of the first biased word
        bias_token_idx = entry["pre_tok_label_ids"].to(dtype=torch.int).flatten().tolist().index(1)

        # the pre_ids were tokenized with the standard BERT Tokenizer
        outputs = model(entry["pre_ids"])[0]
        #TODO: check how we index into the bias_token_idx for the word

        embeddings.append(embedding)

    # num_entries, dim
    tensor_embeddings = torch.stack(embeddings)
    return tensor_embeddings
