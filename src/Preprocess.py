"""
@file: Preprocess.py
@time: 2019/12/28
@desc: 。
"""

from stanfordcorenlp import StanfordCoreNLP
import time


path = ''
data_path = path + 'train1.txt'
nlp_path = ''
nlp_path = nlp_path + 'stanford-corenlp-full-2018-02-27''
nlp = StanfordCoreNLP(nlp_path)

rst = open(path + 'train1.txt', 'w', encoding='utf-8')
with open(data_path, 'r', encoding='utf-8') as f:
    data = f.readlines()
    for text in data:
        print(text)
        if text != "\n":
            fenci = nlp.word_tokenize(text)
            sen = ' '.join(fenci)
            rst.write(sen + '\n')
        else:
            rst.write('\n')

rst.close()
