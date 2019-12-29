# Seq2Seq example by pytorch
--------
This demo contains tutorials covering implementing sequence-to-sequence (seq2seq) model suing Pytorch 1.3(CUDA8.0 version) and TorchText 0.4 on Python 3.6.

## 1. Brief

A sequence-to-sequence model implemented by pytorch and torchtext. This model was designed to translate python into cpp in grammar, however, the logic error and lacking of datasets blocked the original objective.
This model has been diverted into a translator which translates English into Chinese.

## 2. Getting Started

We assume that Python3 has been installed as basis on your system.

Or if not, please check [THIS](https://www.anaconda.com/download)

We highly recommand you to check [THIS](https://pytorch.org/) in order to download a proper version of Pytorch.

`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

or

`pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html`

in which torchvision has been installed as well.

Torchtext installation as below, or you may like to check [THIS](https://pypi.org/project/torchtext/).

`pip install torchtext`

The environment has been settled.

## 3. Dataset
--------

Whole datasets contains 3 parts including a training one named as train1.txt, a testing one named as test1.txt and a valid one named as valid1.txt.

Textual data in each one of them is formed as:
An English word .    一个 汉语 词汇 。

Chinese sentences in original datasets were continuous, however we have to separate every word and punctuation in these sentences with spaces as preprocessing. The slice tool we used here is with stanfordcorenlp, links and references could be found after.

Original form: ['An English word.'] ['一个汉语词汇。']

After sliced: ['An','English','word','.'] ['一个','汉语','词汇','。']


### 3.1 Preparation

The source language contains English sentences that are continuous with spaces between words and a full stop at the end, and Chinses sentences that are continuous with only a full stop at the end. We need to seperate all elements by spaces (all punctuations included), here we would like to use stanfordcorenlp as slice tool to preprocess the raw data.

The source code of Stanford CoreNLP(SCN) is realized in Java, which provides the server mode for interaction. stanfordcorenlp is a Python toolkit that encapsulates SCN. Stanford officially released the Python version, which can be installed directly. For details, please check the [link](https://stanfordnlp.github.io/stanfordnlp/) in Reference.

### 3.2 Setup

StanfordNLP supports Python 3.6 or later. We strongly recommend that you install StanfordNLP from PyPI. If you already have pip or anaconda installed, simply run

`pip install stanfordnlp`

`conda install stanfordnlp`

this should also help resolve all of the dependencies of StanfordNLP, for instance PyTorch 1.0.0 or above.
Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of StanfordNLP and training your own models. For this option, run

`git clone git@github.com:stanfordnlp/stanfordnlp.git`

`cd stanfordnlp`

`pip install -e`

Check out [tutorial](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#getting-started) for more details.

For Chinese language processing, you need additional 2 packages in [HERE](https://nlp.stanford.edu/software/corenlp-backup-download.html), the first one is

`stanford-corenlp-full-2018-02-27.zip`

and the second one is

`stanford-chinese-corenlp-2018-02-27-models.jar`

Run `Preprocess.py` to seperate sentences into words.

Filling up `path = ''` with the path of unprocessed file and `nlp_path = ''` with the path of the unzipped StanfordNLP package.


## 4. Model
--------
Model_attention.py contains the encoder(attn_Encoder) and the decoder(attn_Decoder) functions which has formed the RNN structures of Seq2Seq model, along with the attention model.

### 4.1 Encoder

The original RNN in referenced paper was built as a 4 layers one-way LSTM, for training time we shrink it to a 2-layer GRU structure.

`class attn_Encoder(nn.Module):`

`    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):`

The inputs of class initialization contain input dimension, embedding dimension, hidden dimension, layer numbers and dropout rate.

The input dimension equals to the vocabulary length of source language sentences, other parameters could be found and settled in Paras.py. 

`    def forward(self, src):`

The input of objects is a batch of data of source language, which will be realized in Func train in `Train.py`.

`       return outputs, hidden`

The outputs of encoder are `outputs` which represent fixed vectors of the input sequences, and `hidden` which are the hidden states.

In forward function, the source language scr is designed to be transformed into dense vectors by embedding layer, and these words are embedded and passed to LSTM then.


### 4.2 Attention

The attention model helps the Encoder to encode sequences into contextual vectors based on steps, and decode those encoded vectors differently.

`class Attention(nn.Module):`

`    def __init__(self, enc_hid_dim, dec_hid_dim):`

`    def forward(self, hidden, encoder_outputs):`

The attention model is to bridges the encoder and decoder so it needs the hidden dimension of both encoder and decoder. Also its inputs contain hidden states and outputs of encoder.

`    return F.softmax(attention, dim=1)`

The output of Attention model is a weight vector of matching degree.

### 4.3 Decoder

A same 2-layer-deep LSTM as the Encoder. 

`class attn_Decoder(nn.Module):`

`    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):`

The decoder function is initialized with output dimension, embedding dimension, hidden dimension, layer numbers, dropout rate and weights from attention model.

`    def forward(self, input, hidden, encoder_outputs):`

The `input` is actually a batch of target language, `hidden` and `encoder_outputs` are results from the Encoder. 

`    return prediction, output, hidden.squeeze(0)`

The outputs of Decoder are `prediction` which is a demo test of a single input sentence, `output` and `hidden` are the final results of Decoder.

As a comparison, we have retained the original Encoder and Decoder functions without attention model which could be realized in `Model.py`.

## 5. Seq2Seq network
--------

`class attn_Seq2Seq(nn.Module):`

`    def __init__(self, encoder, decoder, device):`

`    def forward(self, src, trg, teacher_forcing_ratio=0.5):`


The inputs of attention seq2seq model are source language and target language sequences.

`    return outputs`

The output is the corresponding sequence of input.

## 6. Training
--------

Function `train(model, iterator, optimizer, criterion, clip):` takes source language and target language as training data, optimizer and criterion as input.

And returns to loss values as results.

`return epoch_loss / len(iterator)`

## 7. Testing
--------

In order to test the performance of trained model we define a translate function and calculate the PPL at the same time. Inputing the source sentence manually which you need to translate, and test PPL value shows the performance of the trained model theoretically.

`translate(sentence, model, device, SRC, TRG):`

The input of `sentence` is the testing sentence you want to translate.

`    sentence = Test.greedy_search(sentence, model, device, TRG)`

`    return sentence`

And returns to a translated sentence.

## 8. Running program

Running this program in `main_attn.py`, there are several places you may wanna change depending on your own sets.

`train_path = ""`

`valid_path = ""`

`test_path = ""`

Filling up with your own direct paths of data files.

`sentence = ''`

Filling up with any English sentence you wanna translate (suggesting to find a sentence in test1.txt or valid1.txt).

## 9. Testing results
--------

### 9.1 Theoretical criterion

Our hardware testing environment is based on Win10 OS, i7 9-Gen, 16G RAM and RTX2070 GPU. Training time and results has been list in Result.txt, and the testing result of model verification is as followed:

| Test Loss: 2.939 | Test PPL:  18.890 |
| Test Loss: 3.207 | Test PPL:  24.711 |(baseline)

PPL(Preplexity) is a common criterion of language model, its basic idea is that it is better to give a higher probabiliby value to sentences in test sets. When the language model is trained and the sentence in the test set is normal, the higher probability in the test set, the better model is trained. From the definition and formular of PPL, we could know that a better language model should keep a lower value of PPL.

Our average test PPL is around 19 which is much smaller than the [baseline](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) of the paper in Reference.

### 9.2 Practical criterion

A more straightforward way to test the trained model is checking the performance of model in practical applications(spelling check or machine translation). The function `Translate.translate` is to check the translation result of a single English sentence, which is treated as an intuitive approach in human habits.


## Reference
Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[J]. Advances in NIPS, 2014.
https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
https://stanfordnlp.github.io/stanfordnlp/
https://github.com/bentrevett/pytorch-seq2seq
https://pytorch-cn.readthedocs.io/zh/latest/
https://blog.csdn.net/leo_95/article/details/87708267#Field_7

