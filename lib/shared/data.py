__author__ = 'Reid Pryzant'
'''
File used for data pre-processing and tokenization. Contains util functions
and constants.
'''

import os
import pickle
from tqdm import tqdm
from simplediff import diff
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from random import shuffle
import random
from collections import defaultdict
import csv

# from https://spacy.io/api/annotation#section-dependency-parsing
RELATIONS = [
    'det', 'amod', 'nsubj', 'prep', 'pobj', 'ROOT',
    'attr', 'punct', 'advmod', 'compound', 'acl', 'agent',
    'aux', 'ccomp', 'dobj', 'cc', 'conj', 'appos', 'nsubjpass',
    'auxpass', 'poss', 'nummod', 'nmod', 'relcl', 'mark',
    'advcl', 'pcomp', 'npadvmod', 'preconj', 'neg', 'xcomp',
    'csubj', 'prt', 'parataxis', 'expl', 'case', 'acomp', 'predet',
    'quantmod', 'dep', 'oprd', 'intj', 'dative', 'meta', 'csubjpass',
    '<UNK>'
]
REL2ID = {x: i for i, x in enumerate(RELATIONS)}

# from https://spacy.io/api/annotation#section-pos-tagging
POS_TAGS = [
    'DET', 'ADJ', 'NOUN', 'ADP', 'NUM', 'VERB', 'PUNCT', 'ADV',
    'PART', 'CCONJ', 'PRON', 'X', 'INTJ', 'PROPN', 'SYM',
    '<UNK>'
]
POS2ID = {x: i for i, x in enumerate(POS_TAGS)}

# 0: shared 1: edited
EDIT_TYPE2ID = {'0': 0, '1': 1, 'mask': 2}

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def get_tok_labels(s_diff):
    pre_tok_labels = []
    post_tok_labels = []
    for tag, chunk in s_diff:
        if tag == '=':
            pre_tok_labels += [0] * len(chunk)
            post_tok_labels += [0] * len(chunk)
        elif tag == '-':
            pre_tok_labels += [1] * len(chunk)
        elif tag == '+':
            post_tok_labels += [1] * len(chunk)
        else:
            pass

    return pre_tok_labels, post_tok_labels


def noise_seq(seq, drop_prob=0.25, shuf_dist=3, drop_set=None, keep_bigrams=False):
    # from https://arxiv.org/pdf/1711.00043.pdf
    def perm(i):
        return i[0] + (shuf_dist + 1) * np.random.random()

    if drop_set == None:
        dropped_seq = [x for x in seq if np.random.random() > drop_prob]
    else:
        dropped_seq = [x for x in seq if not (x in drop_set and np.random.random() < drop_prob)]

    if keep_bigrams:
        i = 0
        original = ' '.join(seq)
        tmp = []
        while i < len(dropped_seq)-1:
            if ' '.join(dropped_seq[i : i+2]) in original:
                tmp.append(dropped_seq[i : i+2])
                i += 2
            else:
                tmp.append([dropped_seq[i]])
                i += 1

        dropped_seq = tmp

    # global shuffle
    if shuf_dist == -1:
        shuffle(dropped_seq)
    # local shuffle
    elif shuf_dist > 0:
        dropped_seq = [x for _, x in sorted(enumerate(dropped_seq), key=perm)]
    # shuf_dist of 0 = no shuffle

    if keep_bigrams:
        dropped_seq = [z for y in dropped_seq for z in y]

    return dropped_seq

def get_sentences_from_dataset(data_path):
    '''
    Returns tokens for a set of small labeled data examples.
    '''
    toks_dict = {}

    for i, (line) in enumerate(tqdm(open(data_path))):
        parts = line.strip().split('\t')
        # if there pos/rel info
        if len(parts) == 7:
            [revid, pre, post, _, _, pos, rels] = parts
        # no pos/rel info
        elif len(parts) == 5:
            [revid, pre, post, _, _] = parts
        # broken line
        else:
            continue

        tokens = pre.strip().split()
        toks_dict[i] = tokens

    return toks_dict


def get_examples(dataset_params, data_path, tok2id, max_seq_len, noise=False,
                 add_del_tok=False, categories_path=None, convert_to_tensors=True,
                 no_bias_type_labels=False):

    global REL2ID
    global POS2ID
    global EDIT_TYPE2ID

    if dataset_params['drop_words'] is not None:
        drop_set = set([l.strip() for l in
                   open(dataset_params['drop_words'])])
    else:
        drop_set = None

    def pad(id_arr, pad_idx):
        return id_arr + ([pad_idx] * (max_seq_len - len(id_arr)))

    skipped = 0
    out = defaultdict(list)
    if categories_path is not None:
        category_fp = open(categories_path)
        next(category_fp) # ignore header
        revid2topic = {
            l.strip().split(',')[0]: [float(x) for x in l.strip().split(',')[1:]]
            for l in category_fp
        }

    #NOTE: we require that the TSV begins with some header
    data_file = open(data_path)
    header_line = next(data_file).strip().split('\t')
    assert header_line[0] == 'id', "data file required to contain a header."

    for i, (line) in enumerate(tqdm(data_file)):

        parts = line.strip().split('\t')

        if no_bias_type_labels:
            if len(parts) == 5:
                [revid, pre, post, pos, rels] = parts
            elif len(parts) == 7:
                [revid, pre, post, _, _, pos, rels] = parts
            else:
                skipped += 1
                continue
        else:
            if len(parts) == 7:
                [revid, pre, post, pos, rels, epistemological, framing] = parts
            elif len(parts) == 9:
                [revid, pre, post, _, _, pos, rels, epistemological, framing] = parts
            else:
                skipped += 1
                continue

        # break up tokens
        tokens = pre.strip().split()
        post_tokens = post.strip().split()
        rels = rels.strip().split()
        pos = pos.strip().split()

        # get diff + binary diff masks
        tok_diff = diff(tokens, post_tokens)
        pre_tok_labels, post_tok_labels = get_tok_labels(tok_diff)

        # make sure everything lines up
        if len(tokens) != len(pre_tok_labels) \
            or len(tokens) != len(rels) \
            or len(tokens) != len(pos) \
            or len(post_tokens) != len(post_tok_labels):
            skipped += 1
            continue

        # leave room in the post for start/stop and possible category/class token
        if len(tokens) > max_seq_len - 1 or len(post_tokens) > max_seq_len - 1:
            skipped += 1
            continue

        # category info if provided
        # TODO -- if provided but not in diyi's data, we fill with random...is that ok?
        if categories_path is not None and revid in revid2topic:
            categories = revid2topic[revid]
        else:
            categories = np.random.uniform(size=43)   # 43 = number of categories
            categories = categories / sum(categories) # normalize

        if dataset_params['category_input']:
            category_id = np.argmax(categories)
            tokens = ['[unused%d]' % category_id] + tokens
            pre_tok_labels = [EDIT_TYPE2ID['mask']] + pre_tok_labels
            post_tok_labels = [EDIT_TYPE2ID['mask']] + post_tok_labels

        # add start + end symbols to post in/out
        post_input_tokens = ['行'] + post_tokens
        post_output_tokens = post_tokens + ['止']

        # shuffle + convert to ids + pad
        try:
            if noise:
                pre_toks = noise_seq(
                    tokens[:],
                    drop_prob=dataset_params['noise_prob'],
                    shuf_dist=dataset_params['shuf_dist'],
                    drop_set=drop_set,
                    keep_bigrams=dataset_params['keep_bigrams'])
            else:
                pre_toks = tokens

            pre_ids = pad([tok2id[x] for x in pre_toks], 0)
            post_in_ids = pad([tok2id[x] for x in post_input_tokens], 0)
            post_out_ids = pad([tok2id[x] for x in post_output_tokens], 0)
            pre_tok_label_ids = pad(pre_tok_labels, EDIT_TYPE2ID['mask'])
            if 1 not in pre_tok_label_ids:
                skipped += 1
                continue

            post_tok_label_ids = pad(post_tok_labels, EDIT_TYPE2ID['mask'])
            rel_ids = pad([REL2ID.get(x, REL2ID['<UNK>']) for x in rels], 0)
            pos_ids = pad([POS2ID.get(x, POS2ID['<UNK>']) for x in pos], 0)
        except KeyError:
            skipped += 1
            continue

        input_mask = pad([0] * len(tokens), 1)
        pre_len = len(tokens)

        # Adding label encodings --> 0: epistemological; 1:framing
        if not no_bias_type_labels:
            try:
                bias_label = 0 if int(epistemological) else 1
                if bias_label == 1:
                    assert int(framing), "Processing error: both epistemological and framing labels are true."
            except KeyError:
                continue

        out['pre_ids'].append(pre_ids)
        out['masks'].append(input_mask)
        out['pre_lens'].append(pre_len)
        out['post_in_ids'].append(post_in_ids)
        out['post_out_ids'].append(post_out_ids)
        out['pre_tok_label_ids'].append(pre_tok_label_ids)
        out['post_tok_label_ids'].append(post_tok_label_ids)
        out['rel_ids'].append(rel_ids)
        out['pos_ids'].append(pos_ids)
        out['categories'].append(categories)
        out['index'].append(i) # can do some hash thing
        if not no_bias_type_labels:
            out['bias_label'].append(bias_label)

    if convert_to_tensors:
        out['pre_ids'] = torch.tensor(out['pre_ids'], dtype=torch.long)
        out['masks'] = torch.tensor(out['masks'], dtype=torch.uint8) # byte for masked_fill()
        out['pre_lens'] = torch.tensor(out['pre_lens'], dtype=torch.long)
        out['post_in_ids'] = torch.tensor(out['post_in_ids'], dtype=torch.long)
        out['post_out_ids'] = torch.tensor(out['post_out_ids'], dtype=torch.long)
        out['pre_tok_label_ids'] = torch.tensor(out['pre_tok_label_ids'], dtype=torch.float)  # for compartin to enrichment stuff
        out['post_tok_label_ids'] = torch.tensor(out['post_tok_label_ids'], dtype=torch.float)  # for loss multiplying
        out['rel_ids'] = torch.tensor(out['rel_ids'], dtype=torch.long)
        out['pos_ids'] = torch.tensor(out['pos_ids'], dtype=torch.long)
        out['categories'] = torch.tensor(out['categories'], dtype=torch.float)
        out['index'] = torch.tensor(out['index'], dtype=torch.int32)
        if not no_bias_type_labels:
            out['bias_label'] = torch.tensor(out['bias_label'], dtype=torch.int16)


    return out


def get_dataloader(data_path, tok2id, batch_size, params, target_labels_path=None,
                   pickle_path=None, test=False, noise=False, add_del_tok=False,
                   categories_path=None, sort_batch=True):
    ''' Params is intemediary task specific dictionary'''

    def collate(data):
        if target_labels_path is not None:
            if sort_batch:
                # sort by length for packing/padding
                data.sort(key=lambda x: x[2], reverse=True)
            # group by datatype
            [
                src_id, src_mask, src_len,
                post_in_id, post_out_id,
                pre_tok_label, post_tok_label,
                rel_ids, pos_ids, categories, bias_labels, indices
            ] = [torch.stack(x) for x in zip(*data)]

            # cut off at max len of this batch for unpacking/repadding
            max_len = max(src_len)
            data = [
                src_id[:, :max_len], src_mask[:, :max_len], src_len,
                post_in_id[:, :max_len+10], post_out_id[:, :max_len+10],    # +10 for wiggle room
                pre_tok_label[:, :max_len], post_tok_label[:, :max_len+10], # +10 for post_toks_labels too (it's just gonna be matched up with post ids)
                rel_ids[:, :max_len], pos_ids[:, :max_len], categories, bias_labels, indices
            ]
        else:
            if sort_batch:
                # sort by length for packing/padding
                data.sort(key=lambda x: x[2], reverse=True)
            # group by datatype
            [
                src_id, src_mask, src_len,
                post_in_id, post_out_id,
                pre_tok_label, post_tok_label,
                rel_ids, pos_ids, categories, indices
            ] = [torch.stack(x) for x in zip(*data)]

            # cut off at max len of this batch for unpacking/repadding
            max_len = max(src_len)
            data = [
                src_id[:, :max_len], src_mask[:, :max_len], src_len,
                post_in_id[:, :max_len+10], post_out_id[:, :max_len+10],    # +10 for wiggle room
                pre_tok_label[:, :max_len], post_tok_label[:, :max_len+10], # +10 for post_toks_labels too (it's just gonna be matched up with post ids)
                rel_ids[:, :max_len], pos_ids[:, :max_len], categories, indices
            ]

        return data

    examples = get_examples(
        params,
        data_path=data_path,
        target_labels_path=target_labels_path,
        tok2id=tok2id,
        max_seq_len=params['general_model_params']['max_seq_len'],
        noise=noise,
        add_del_tok=add_del_tok,
        categories_path=categories_path)

    if target_labels_path is not None:
        data = TensorDataset(
            torch.tensor(examples['pre_ids'], dtype=torch.long),
            torch.tensor(examples['pre_masks'], dtype=torch.uint8), # byte for masked_fill()
            torch.tensor(examples['pre_lens'], dtype=torch.long),
            torch.tensor(examples['post_in_ids'], dtype=torch.long),
            torch.tensor(examples['post_out_ids'], dtype=torch.long),
            torch.tensor(examples['pre_tok_label_ids'], dtype=torch.float),  # for compartin to enrichment stuff
            torch.tensor(examples['post_tok_label_ids'], dtype=torch.float),  # for loss multiplying
            torch.tensor(examples['rel_ids'], dtype=torch.long),
            torch.tensor(examples['pos_ids'], dtype=torch.long),
            torch.tensor(examples['categories'], dtype=torch.float),
            torch.tensor(examples['bias_label'], dtype=torch.int16),
            torch.tensor(examples['index'], dtype=torch.int16))
    else:
        data = TensorDataset(
            torch.tensor(examples['pre_ids'], dtype=torch.long),
            torch.tensor(examples['pre_masks'], dtype=torch.uint8), # byte for masked_fill()
            torch.tensor(examples['pre_lens'], dtype=torch.long),
            torch.tensor(examples['post_in_ids'], dtype=torch.long),
            torch.tensor(examples['post_out_ids'], dtype=torch.long),
            torch.tensor(examples['pre_tok_label_ids'], dtype=torch.float),  # for compartin to enrichment stuff
            torch.tensor(examples['post_tok_label_ids'], dtype=torch.float),  # for loss multiplying
            torch.tensor(examples['rel_ids'], dtype=torch.long),
            torch.tensor(examples['pos_ids'], dtype=torch.long),
            torch.tensor(examples['categories'], dtype=torch.float),
            torch.tensor(examples['index'], dtype=torch.int16))

    dataloader = DataLoader(
        data,
        sampler=(SequentialSampler(data) if test else RandomSampler(data)),
        collate_fn=collate,
        batch_size=batch_size)

    return dataloader, len(examples['pre_ids'])
