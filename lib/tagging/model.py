__author__ = 'Reid Pryzant'

from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertSelfAttention, BertSelfOutput, BertModel
import pytorch_pretrained_bert.modeling as modeling
import torch
import torch.nn as nn
import numpy as np
import copy
import lib.tagging.features as features
from lib.shared.constants import CUDA

class ConcatCombine(nn.Module):
    def __init__(self, hidden_size, feature_size, out_size, layers,
            dropout_prob, small=False, pre_enrich=False, activation=False,
            include_categories=False, category_emb=False,
            add_category_emb=False):
        super(ConcatCombine, self).__init__()

        self.include_categories = include_categories
        self.add_category_emb = add_category_emb
        if include_categories:
            if category_emb and not add_category_emb:
                feature_size *= 2
            elif not category_emb:
                feature_size += 43

        if layers == 1:
            self.out = nn.Sequential(
                nn.Linear(hidden_size + feature_size, out_size),
                nn.Dropout(dropout_prob))
        elif layers == 2:
            waist_size = min(hidden_size, feature_size) if small else max(hidden_size, feature_size)
            if activation:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size + feature_size, waist_size),
                    nn.Dropout(dropout_prob),
                    nn.ReLU(),
                    nn.Linear(waist_size, out_size),
                    nn.Dropout(dropout_prob))
            else:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size + feature_size, waist_size),
                    nn.Dropout(dropout_prob),
                    nn.Linear(waist_size, out_size),
                    nn.Dropout(dropout_prob))
        if pre_enrich:
            if activation:
                self.enricher = nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    nn.ReLU())
            else:
                self.enricher = nn.Linear(feature_size, feature_size)
        else:
            self.enricher = None
        # manually set cuda because module doesn't see these combiners for bottom
        if CUDA:
            self.out = self.out.cuda()
            if self.enricher:
                self.enricher = self.enricher.cuda()

    def forward(self, hidden, features, categories=None):
        if self.include_categories:
            categories = categories.unsqueeze(1)
            categories = categories.repeat(1, features.shape[1], 1)
            if self.add_category_emb:
                features = features + categories
            else:
                features = torch.cat((features, categories), -1)

        if self.enricher is not None:
            features = self.enricher(features)

        return self.out(torch.cat((hidden, features), dim=-1))

class BertForMultitaskWithFeaturesOnTop(PreTrainedBertModel):
    """ stick the features on top of the model """
    def __init__(self, config, params, cls_num_labels=2, tok_num_labels=2, tok2id=None):
        super(BertForMultitaskWithFeaturesOnTop, self).__init__(config)

        self.params = params
        self.bert = BertModel(config)

        self.featurizer = features.Featurizer(
            tok2id,
            lexicon_feature_bits = params['lexicon_feature_bits'],
            params = params)
        nfeats = 90 if params['lexicon_feature_bits'] == 1 else 118

        self.tok_classifier = ConcatCombine(
            config.hidden_size,
            nfeats,
            tok_num_labels,
            params['combiner_layers'],
            config.hidden_dropout_prob,
            params['small_waist'],
            pre_enrich = params['pre_enrich'],
            activation = params['activation_hidden'],
            include_categories = params['concat_categories'],
            category_emb = params['category_emb'],
            add_category_emb = params['add_category_emb'],)

        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, cls_num_labels)

        self.category_emb = params['category_emb']
        if params['category_emb']:
            self.category_embeddings = nn.Embedding(43, nfeats)

        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
        labels=None, rel_ids=None, pos_ids=None, categories=None):

        global CUDA

        features = self.featurizer.featurize_batch(
            input_ids.detach().cpu().numpy(),
            rel_ids.detach().cpu().numpy(),
            pos_ids.detach().cpu().numpy(),
            padded_len=input_ids.shape[1])
        features = torch.tensor(features, dtype=torch.float)
        if CUDA:
            features = features.cuda()

        sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        pooled_output = self.cls_dropout(pooled_output)
        cls_logits = self.cls_classifier(pooled_output)

        if self.params['category_emb']:
            categories = self.category_embeddings(
                categories.max(-1)[1].type(
                    'torch.cuda.LongTensor' if CUDA else 'torch.LongTensor'))

        tok_logits = self.tok_classifier(sequence_output, features, categories)

        return cls_logits, tok_logits
