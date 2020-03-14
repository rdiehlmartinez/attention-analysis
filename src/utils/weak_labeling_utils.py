__author__ = 'Richard Diehl Martinez'

'''
Util file that stores useful functions and variables used for the weak
labeling functions.
'''

import numpy as np
import torch
from .data_utils import get_words_and_indices, get_tok2id, get_id2tok

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

    assert("lexicon_dir" in dataset_params.keys()), \
        "Lexicon directory path must be specified in the dataset parameters"

    tok2id = get_tok2id(dataset_params)
    return Featurizer(tok2id, params=dataset_params)

def extract_marta_features(dataset, featurizer, bias_key="pre_tok_label_ids"):
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

    from tqdm import tqdm_notebook as tqdm

    full_features = []
    for entry in tqdm(dataset):
        features = featurizer.features(entry["pre_ids"].tolist(),
                                       entry["rel_ids"].tolist(),
                                       entry["pos_ids"].tolist())

        # Figuring out what the index is of the first biased word
        bias_idx = entry[bias_key].to(dtype=torch.int).flatten().tolist().index(1)

        full_features.append(features[bias_idx, :])
    # num_entries, dim
    tensor_features = torch.tensor(np.stack(full_features), dtype=torch.float32)
    return tensor_features

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
        * tensor_embeddings (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    from tqdm import tqdm_notebook as tqdm

    embeddings = []
    id2tok = get_id2tok(dataset_params)

    # loading in the GloVe dictionary
    glove_dict = load_glove_dict(dataset_params)

    for entry in tqdm(dataset):
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
            embedding = torch.zeros(dataset_params['glove_embedding_dim'], dtype=torch.float32)

        embeddings.append(embedding)

    # num_entries, dim
    tensor_embeddings = torch.stack(embeddings)
    return tensor_embeddings


def get_pos_features(dataset_params, dataset):
    '''
    Iterates over a dataset and for the predicted biased word generates the pos
    tag for that word. We convert the predicted biased word to a one-hot vector,
    and then combine together all of the vectors into one matrix.

    Args:
        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * pos_matrix (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    import nltk
    from sklearn.feature_extraction import DictVectorizer
    from tqdm import tqdm_notebook as tqdm

    pos_tags = []
    id2tok = get_id2tok(dataset_params)

    for entry in tqdm(dataset):
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
    pos_matrix = torch.tensor(pos_matrix, dtype=torch.float32)

    return pos_matrix


def get_pos_features_multi_dataset(dataset_params, datasets):
    '''
    Iterates over a list of datasets and for the predicted biased word generates the pos
    tag for that word. The reason we have this as a separate function from the
    version using only one dataset is that the for each dataset the POS generates
    a different amount of tags, depending on how many unique POS tags occur in the
    dataset.

    Args:
        * dataset_params (dict): A dictionary of parameter values specifying
            configurations for the dataset.
        * datasets ([Experimentdataset]): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * pos_matrix ([tensor]): a list of tensors of the same number of dimensions
            as the other tensors in the dataset - for each of the datasets
            we pass in.
    '''
    import nltk
    from sklearn.feature_extraction import DictVectorizer
    from tqdm import tqdm_notebook as tqdm

    pos_tags = []
    id2tok = get_id2tok(dataset_params)

    for dataset in datasets:
        for entry in tqdm(dataset):
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
    pos_matrices = []
    curr_idx = 0
    for dataset in datasets:
        curr_len = len(dataset)
        curr_pos_matrix = torch.tensor(pos_matrix[curr_idx: curr_idx+curr_len, :], dtype=torch.float32)
        curr_idx += curr_len
        pos_matrices.append(curr_pos_matrix)

    return pos_matrices

def get_bert_features(dataset):
    '''
    Iterates over a dataset and for the predicted biased word generates the
    contextualized BERT embedding for that word. For now, we only use the
    last hidden state to extract the BERT embedding.

    Args:
        * dataset (Experimentdataset): defined in src/dataset
            - We require each entry in the dataset to contain information/key
                values for the 1) pre_ids, 2) rel_ids, 3) pos_ids

    Returns:
        * tensor_embeddings (tensor): a tensor of the same number of dimensions as the other tensors
            in the dataset - features contains the extracted features.
    '''
    from pytorch_pretrained_bert import BertModel
    from tqdm import tqdm_notebook as tqdm

    model = BertModel.from_pretrained('bert-base-uncased')

    embeddings = []

    for entry in tqdm(dataset):
        # Figuring out what the index is of the first biased word; and of
        # the pad tokens
        pad_token_idx = entry["pre_tok_label_ids"].to(dtype=torch.int).flatten().tolist().index(2)
        bias_token_idx = entry["pre_tok_label_ids"].to(dtype=torch.int).flatten().tolist().index(1)

        # the pre_ids were tokenized with the standard BERT Tokenizer
        outputs = model(entry["pre_ids"][:pad_token_idx].unsqueeze(0))
        hidden_states = outputs[0]
        last_hidden_state = hidden_states[-1]

        # casting to Float and detaching from BertModel
        embedding = torch.tensor(last_hidden_state[:, bias_token_idx, :], dtype=torch.float32).detach().squeeze()
        embeddings.append(embedding)

    # num_entries, dim
    tensor_embeddings = torch.stack(embeddings)
    return tensor_embeddings

def generate_snorkel_matrix(predictions_list):
    '''
    Given a set of predictions genreated by labeling functions joins these
    predictions together into one label matrix that can be processed by Metal Snorkel.

    args:
        * predictions_list ([List]): a list of the predictions
            generates by the weak labeling functions. Predictions are a list of
            int values.
    returns:
        * lf_matrix (np.matrix)
    '''
    predictions_list = [np.expand_dims(np.array(predictions), axis=1) for predictions in predictions_list]
    lf_matrix = np.concatenate(predictions_list, axis=-1) + 1
    return lf_matrix
