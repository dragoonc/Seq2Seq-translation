#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=UTF-8 

"""
@file: Model.py
@time: 2019/11/30
@desc: Reading datasets, define Encoding and Decoding network
"""

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data
from torchtext.data import Iterator
import torch.nn.functional as F

class Dataset(data.Dataset):
    def __init__(self, path, src_field, trg_field, **kwargs):
        fields = [("src", src_field), ("trg", trg_field)]
        examples = []
        print('loading dataset from {}'.format(path))
        with open(path, encoding="utf-8") as f:
            # special
            for line in f.readlines():
                src = line.split("\t")[0]
                trg = line.split("\t")[1].replace("\n", "")
                trg = trg[::-1]
                examples.append(data.Example.fromlist([src, trg], fields=fields))

        print('size of dataset in {} is : {}'.format(path, len(examples)))
        super(Dataset, self).__init__(examples, fields, **kwargs)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim  # 输入encoder的one-hot向量的维度
        self.emb_dim = emb_dim  # embedding层的维度，该层将one-hot向量转换成dense向量
        self.hid_dim = hid_dim  # 隐状态的维度
        self.n_layers = n_layers  # RNN的层数
        self.dropout = dropout  # 正则化参数，防止过拟合
        self.embedding = nn.Embedding(input_dim, emb_dim)  # 为什么要加这个embedding层呢？直接用emb_dim不行吗
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # self.fc=nn.Linear(enc_hid_dim*2,dec_hid_dim)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src sentence length, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # print(hidden.shape)
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # 在第一维增加一个维度，将input的维度变为（1，batch size）
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))  # dropout不是说output跟input的形状一样吗？
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
