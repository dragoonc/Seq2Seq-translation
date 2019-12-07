#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=UTF-8 

"""
@file: main.py
@time: 2019/12/3
@desc: Main program, data preprocessing, modeling, model training and testing.
"""
import math
import time
import sys
import os
import random
import torch
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import Model
import Seq2Seq
import Train
import Test
import Translate
import Paras
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  # default algorithm, ensuring the fixed outputs when feeding the same inputs

### class Dataset



SRC = data.Field(init_token=Paras.BOS_WORD, eos_token=Paras.EOS_WORD, lower=True)
TRG = data.Field(init_token=Paras.BOS_WORD, eos_token=Paras.EOS_WORD, lower=True)

train_path = "train1.txt"
valid_path = "valid1.txt"
test_path = "test1.txt"
train_data = Model.Dataset(train_path, src_field=SRC, trg_field=TRG)
valid_data = Model.Dataset(valid_path, src_field=SRC, trg_field=TRG)
test_data = Model.Dataset(test_path, src_field=SRC, trg_field=TRG)

SRC.build_vocab(train_data, min_freq=2)  # 建立词汇表的时候需要valid和test两个数据集内容
TRG.build_vocab(train_data, min_freq=2)

# if  args.use_gpu and torch.cuda.is_available():
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device, sort_key=lambda x: (len(x.src), len(x.trg)),sort_within_batch=True, repeat=False)


### Encoder
### Decoder
### Seq2Seq

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)

enc = Model.Encoder(INPUT_DIM, Paras.ENC_EMB_DIM, Paras.HID_DIM, Paras.N_LAYERS, Paras.ENC_DROPOUT)
dec = Model.Decoder(OUTPUT_DIM, Paras.DEC_EMB_DIM, Paras.HID_DIM, Paras.N_LAYERS, Paras.DEC_DROPOUT)

model = Seq2Seq.Seq2Seq(enc, dec, device).to(device)

### init_weights

model.apply(Model.init_weights)

### count_parameters

print(f'The model has {Model.count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

# calculating loss，ignoring loss of padding token
PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

### train
### evaluate
### epoch_time
### data_preprocess
### greedy_search
### translate

N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')
# print("best_valid_loss: ")
# print(best_valid_loss)
for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = Train.train(model, train_iterator, optimizer, criterion, CLIP)
    # print("train_loss: ")
    # print(train_loss)

    valid_loss = Train.evaluate(model, valid_iterator, criterion)

    # print("valid_loss: ")
    # print(valid_loss)
    end_time = time.time()

    epoch_mins, epoch_secs = Test.epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# load the parameters (state_dict) that gave our model the best validation loss and run it the model on the test set.
model.load_state_dict(torch.load('tut1-model.pt'))
sentence = 'Hi.'
result = Translate.translate(sentence, model, device, SRC, TRG)
print(result)
test_loss = Train.evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
