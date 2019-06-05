"""
finetune both models jointly

python joint/train.py \
    --train ../../data/v6/corpus.wordbiased.tag.train \
    --train_categories_file ../../data/v6/corpus.wordbiased.tag.train.categories \
    --test ../../data/v6/corpus.wordbiased.tag.test.categories \
    --pretrain_data ../../data/v6/corpus.unbiased.shuf \
    --extra_features_top --pre_enrich --activation_hidden --category_input --tagging_pretrain_epochs 2 \
    --pretrain_epochs 4 \
    --learning_rate 0.0003 --epochs 2 --hidden_size 10 --train_batch_size 32 --test_batch_size 16 \
    --bert_full_embeddings --debias_weight 1.3 --freeze_tagger --token_softmax --sequence_softmax \
    --working_dir TEST \
    --debug_skip
"""

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import os
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from simplediff import diff
import pickle
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import Counter
import math
import functools

from pytorch_pretrained_bert.modeling import BertEmbeddings
from pytorch_pretrained_bert.optimization import BertAdam


import sys; sys.path.append(".")
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import model as joint_model
import utils as joint_utils



working_dir = ARGS.working_dir
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

with open(working_dir + '/command.sh', 'w') as f:
    f.write('python' + ' '.join(sys.argv) + '\n')

writer = SummaryWriter(ARGS.working_dir)



# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(
    ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab

tok2id['<del>'] = len(tok2id)
print("The len of our vocabulary is {}".format(len(tok2id)))

if ARGS.pretrain_data:
    pretrain_dataloader, num_pretrain_examples = get_dataloader(
        ARGS.pretrain_data,
        tok2id,
        ARGS.train_batch_size,
        ARGS.working_dir + '/pretrain_data.pkl',
        noise=True)

train_dataloader, num_train_examples = get_dataloader(
    ARGS.train,
    tok2id, ARGS.train_batch_size, ARGS.working_dir + '/train_data.pkl',
    categories_path=ARGS.categories_file,
    add_del_tok=ARGS.add_del_tok)
eval_dataloader, num_eval_examples = get_dataloader(
    ARGS.test,
    tok2id, ARGS.test_batch_size, ARGS.working_dir + '/test_data.pkl',
    categories_path=ARGS.categories_file,
    test=True, add_del_tok=ARGS.add_del_tok)



# # # # # # # # ## # # # ## # # TAGGING MODEL # # # # # # # # ## # # # ## # #
# build model
if ARGS.extra_features_top:
    tag_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    tag_model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    tag_model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache')
if CUDA:
    tag_model = tag_model.cuda()

# train or load model
tagging_loss_fn = tagging_utils.build_loss_fn(debias_weight=1.0)

if ARGS.tagger_checkpoint is not None and os.path.exists(ARGS.tagger_checkpoint):
    print('LOADING TAGGER FROM ' + ARGS.tagger_checkpoint)
    tag_model.load_state_dict(torch.load(ARGS.tagger_checkpoint))
    print('...DONE')
else:
    print('TRAINING TAGGER...')
    tagging_optim = tagging_utils.build_optimizer(
        tag_model,
        int((num_train_examples * ARGS.tagging_pretrain_epochs) / ARGS.train_batch_size),
        ARGS.tagging_pretrain_lr)

    # print('INITIAL EVAL...')
    # tag_model.eval()
    # results = tagging_utils.run_inference(
    #     tag_model, eval_dataloader, tagging_loss_fn, tokenizer)
    # writer.add_scalar('tag_eval/tok_loss', np.mean(results['tok_loss']), 0)
    # writer.add_scalar('tag_eval/tok_acc', np.mean(results['labeling_hits']), 0)

    print('TRAINING...')
    for epoch in range(ARGS.tagging_pretrain_epochs):
        print('EPOCH ', epoch)
        tag_model.train()
        losses = tagging_utils.train_for_epoch(
            tag_model, train_dataloader, tagging_loss_fn, tagging_optim)
        writer.add_scalar('tag_train/loss', np.mean(losses), epoch + 1)

        tag_model.eval()
        results = tagging_utils.run_inference(
            tag_model, eval_dataloader, tagging_loss_fn, tokenizer)
        writer.add_scalar('tag_eval/tok_loss', np.mean(results['tok_loss']), epoch + 1)
        writer.add_scalar('tag_eval/tok_acc', np.mean(results['labeling_hits']), epoch + 1)



# # # # # # # # ## # # # ## # # DEBIAS MODEL # # # # # # # # ## # # # ## # #
# bulid model
if ARGS.pointer_generator:
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size
else:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)
if CUDA:
    debias_model = debias_model.cuda()

# train or load model
debias_loss_fn, cross_entropy_loss = seq2seq_utils.build_loss_fn(vocab_size=len(tok2id))

if ARGS.debias_checkpoint is not None and os.path.exists(ARGS.debias_checkpoint):
    print('LOADING DEBIASER FROM ' + ARGS.debias_checkpoint)
    tag_model.load_state_dict(torch.load(ARGS.debias_checkpoint))
    print('...DONE')

elif ARGS.pretrain_data:
    pretrain_optim = optim.Adam(debias_model.parameters(), lr=ARGS.learning_rate)

    print('PRETRAINING...')
    debias_model.train()
    # TODO -- VERIFY THAT TAGGER IS ACTUALLY BEING IGNORED, I.E. TOK DIST IS ALL 0'S
    for epoch in range(ARGS.pretrain_epochs):
        print('EPOCH ', epoch)
        print('TRAIN...')
        losses = seq2seq_utils.train_for_epoch(
            debias_model, pretrain_dataloader, tok2id, pretrain_optim, cross_entropy_loss,
            ignore_enrich=not ARGS.use_pretrain_enrich)
        writer.add_scalar('pretrain/loss', np.mean(losses), epoch)


# # # # # # # # # # # # JOINT MODEL # # # # # # # # # # # # # #

# build model
joint_model = joint_model.JointModel(
    debias_model=debias_model, tagging_model=tag_model)

if CUDA:
    joint_model = joint_model.cuda()

model_parameters = filter(lambda p: p.requires_grad, joint_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('NUM PARAMS: ', params)

joint_optimizer = optim.Adam(
    debias_model.parameters() if ARGS.freeze_tagger else joint_model.parameters(),
    lr=ARGS.learning_rate)

# train model
print('JOINT TRAINING...')
print('INITIAL EVAL...')
joint_model.eval()
hits, preds, golds, srcs = joint_utils.run_eval(
    joint_model, eval_dataloader, tok2id, ARGS.working_dir + '/results_initial.txt',
    ARGS.max_seq_len, ARGS.beam_width)
writer.add_scalar('eval/bleu', seq2seq_utils.get_bleu(preds, golds), 0)
writer.add_scalar('eval/true_hits', np.mean(hits), 0)

for epoch in range(ARGS.epochs):
    print('EPOCH ', epoch)
    print('TRAIN...')
    losses = joint_utils.train_for_epoch(
        joint_model, train_dataloader, joint_optimizer,
        debias_loss_fn, tagging_loss_fn, ignore_tagger=False)
    writer.add_scalar('train/loss', np.mean(losses), epoch + 1)


    print('SAVING...')
    joint_model.save(ARGS.working_dir + '/model_%d.ckpt' % (epoch + 1))

    print('EVAL...')
    joint_model.eval()
    hits, preds, golds, srcs = joint_utils.run_eval(
        joint_model, eval_dataloader, tok2id, ARGS.working_dir + '/results_%d.txt' % epoch,
        ARGS.max_seq_len, ARGS.beam_width)
    writer.add_scalar('eval/bleu', seq2seq_utils.get_bleu(preds, golds), epoch+1)
    writer.add_scalar('eval/true_hits', np.mean(hits), epoch+1)
