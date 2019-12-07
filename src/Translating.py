#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# coding=UTF-8 

"""
@file: Translate.py
@time: 2019/12/3
@desc: Translating the test sample, take single English sentence as input
"""
import Test
import torch
from torch.autograd import Variable
import Paras

def data_preprocess(SRC, sentence):
    return [SRC.vocab.stoi[word] for word in sentence.split(' ')] + [SRC.vocab.stoi[Paras.EOS_WORD]]

def translate(sentence, model, device, SRC, TRG):

    model.eval()
    indexed = data_preprocess(SRC, sentence)

    # 需要加view函数转置sentence矩阵，因为得到的是(batch_size,sen_len),而encoder的输入需要(sen_len,batch_size)
    sentence = Variable(torch.LongTensor([indexed])).view(-1,1)

    sentence = Test.greedy_search(sentence, model, device, TRG)
    return sentence
