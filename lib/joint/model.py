import torch.nn as nn
import torch
from torch.autograd import Variable
from tqdm import tqdm
from lib.shared.constants import CUDA

class JointModel(nn.Module):
    def __init__(self, params, debias_model, tagging_model):
        super(JointModel, self).__init__()

        self.params = params

        self.debias_model = debias_model
        self.tagging_model = tagging_model

        self.token_sm = nn.Softmax(dim=2)
        self.time_sm = nn.Softmax(dim=1)
        self.tok_threshold = nn.Threshold(
            self.params['zero_threshold'] ,
            -10000.0 if self.params['sequence_softmax']  else 0.0)

    def run_tagger(self, pre_id, pre_mask, rel_ids=None, pos_ids=None,
                   categories=None):
        _, tok_logits = self.tagging_model(
            pre_id, attention_mask=1.0 - pre_mask, rel_ids=rel_ids,
            pos_ids=pos_ids, categories=categories)

        tok_probs = tok_logits[:, :, :2]
        if self.params['token_softmax']:
            tok_probs = self.token_sm(tok_probs)
        is_bias_probs = tok_probs[:, :, -1]
        is_bias_probs = is_bias_probs.masked_fill(pre_mask, 0.0)

        if self.params['zero_threshold'] > -10000.0:
            is_bias_probs = self.tok_threshold(is_bias_probs)

        if self.params['sequence_softmax']:
            is_bias_probs = self.time_sm(is_bias_probs)

        return is_bias_probs, tok_logits

    def forward(self,
            # Debias args.
            pre_id, post_in_id, pre_mask, pre_len, tok_dist,
            # Tagger args.
            rel_ids=None, pos_ids=None, categories=None, ignore_tagger=False):
        global ARGS

        if ignore_tagger:
            is_bias_probs = tok_dist
            tok_logits = None
        else:
            is_bias_probs, tok_logits = self.run_tagger(
                pre_id, pre_mask, rel_ids, pos_ids, categories)

        post_log_probs, post_probs = self.debias_model(
            pre_id, post_in_id, pre_mask, pre_len, is_bias_probs)

        return post_log_probs, post_probs, is_bias_probs, tok_logits


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
