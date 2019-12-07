# Seq2Seq example by pytorch
This demo contains tutorials covering implementing sequence-to-sequence (seq2seq) model suing Pytorch 1.1 and TorchText 0.4 on Python 3.6.

-Brief
A sequence-to-sequence model implemented by pytorch and torchtext. This model was designed to translate python into cpp in grammar, however, the logic error and lacking of datasets blocked the original objective.
This model has been diverted into a translator which translates English into Chinese.

-Getting Started
We will pass the installation of PyTorch and TorchText as there are plenty of instructions on the PyTorch website and others.

---------------------------------------------------------------------------------------------------------------------------------------

## Dataset
--------

Whole datasets contains 3 parts including a training one named as train1.txt, a testing one named as test1.txt and a valid one named as valid1.txt.

Textual data in each one of them is formed as:
An English word .    一个 汉语 词汇 。

Chinese sentences in original datasets were continuous, however we have to separate every word and punctuation in these sentences with spaces as preprocessing. The slice tool we used here is with stanfordcorenlp, links and references could be found after.
Original form: ['An English word.'] ['一个汉语词汇。']
After sliced: ['An','English','word','.'] ['一个','汉语','词汇','。']

### Preparation
--------

The source language contains English sentences that are continuous with spaces between words and a full stop at the end, and Chinses sentences that are continuous with only a full stop at the end. We need to seperate all elements by spaces (all punctuations included), here we would like to use stanfordcorenlp as slice tool to preprocess the raw data.

The source code of Stanford CoreNLP(SCN) is realized in Java, which provides the server mode for interaction. stanfordcorenlp is a Python toolkit that encapsulates SCN. Stanford officially released the Python version, which can be installed directly. For details, please check the [link](https://stanfordnlp.github.io/stanfordnlp/) in Reference.

-Setup
--------
StanfordNLP supports Python 3.6 or later. We strongly recommend that you install StanfordNLP from PyPI. If you already have pip or anaconda installed, simply run

`pip install stanfordnlp

`conda install stanfordnlp

this should also help resolve all of the dependencies of StanfordNLP, for instance PyTorch 1.0.0 or above.
Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of StanfordNLP and training your own models. For this option, run

git clone git@github.com:stanfordnlp/stanfordnlp.git
cd stanfordnlp
pip install -e .

-Running StanfordNLP
--------
Getting Started with the neural pipeline
To run your first StanfordNLP pipeline, simply following these steps in your Python interactive interpreter:

>>> import stanfordnlp
>>> stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
>>> nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()

The last command will print out the words in the first sentence in the input string (or Document, as it is represented in StanfordNLP), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its “head”), along with the dependency relation between the words. 

It has also been provided with a multilingual demo script that demonstrates how one uses StanfordNLP in other languages than English, for example Chinese(traditional)

`python demo/pipeline_demo.py -l zh

Check out [tutorial](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#getting-started) for more details.


## Model
--------

### -Encoder
The original RNN in referenced paper was built as a 4 layers one-way LSTM, for training time we shrink it to 2 layers.
Parameters explanation:
    -- input_dim: the dimension of one-hot vector which is supposed to be input into the Encoder, the size - matches the vocabulary's size of original language;
    -- emb_dim: dimension of embedding layer;
    -- hid_dim: dimensions of hidden layers H and C;
    -- dropout: regularizing parameters to prevent over fitting
In forward function, the source language scr is designed to be transformed into dense vectors by embedding layer, and these words are embedded and passed to LSTM then.

### -Decoder
A same 2-layer-deep LSTM as the Encoder. Paras in Decoder are initialized similarly as Encoder except:
    -- output_dim: The dimension of one-hot vector which is regarded as input to the Decoder, the size equals to the vocabulary's size of original language;
    -- Linear layer: same function as in CNN etc. to output the final prediction.
In the forward function of Decoder, we treat the target language trg as input. We utilize unsqueeze() to add a dimension with sentence length of 1 to the input word of target language(the input of source language is a whole sentence), as the result, the processed word is able to be input in embedding layer. Similar to the Encoder, we use an embedding layer and dropout, these embedments are then passed to LSTM along with the hidden state h_n and cell state c_n generated by Encoder layer. Specially note that, in Encoder, we choose a tensor of all 0 as the initial hidden state and cell state. In Decoder, we use the h_n and c_n generated by Decoder as initial hidden state and cell state, which is equivalent to using the context information of the source language in translation.



## Seq2Seq network
--------

Seq2Seq network is designed to combine Encoder and Decoder to realize the following functions:
    -- using source language sentences as input
    -- using Encoder to generate context vector
    -- using Decoder to predict target language sentences
Parameters explanation::
    -- device: put the tensor on GPU. The new version of Pytorch uses the '.to' method to easily move objects to different devices (instead of the previous CPU() or CUDA() methods)
    -- outputs: store all output tensors of Decoder
    -- teacher_forcing_ratio: is a probability to use teacher forcing. When using 'teacher force', the next input of Decoder network is the next character of the target language, on the contrary, the next input of the network is the predicted character. e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time.
In this network, the number of layers(n_layers) and dimension of hidden layer(hid_dim) are equal. However, in other S2S models, it's not always necessary to have the same number of layers or the same size of hidden dimension. For instance, an 2-layer Encoder and 1-layer Decoder, which requires corresponding processing, such as averageing the two context vectors output by Encoder, or only using the context vector of the last layer as input of Decoder.



## Model Training
--------

In training process, we first define paras and weights of Encoder, Decoder and an optimizer. After then we define training and testing functions. The model could be trained when all settled.

### Define model parameters
The embedding dimension(emb_dim) and dropout could be different from Encoder to Decoder, yet the numbers of layer and hidden dimension should be the same. In the original paper, the author initialized all parameters as a uni-distribution between -0.08 to +0.08. We initialize parameters and weights by building a function. When using the '.apply' method, all modules and sub-mods are able to call the function init_weights. '.Adam' optimizer is preferred. When using the Cross Entropy Loss as the loss function, the value of target should be ignored since Pytorch seeks average in batch when calculates the cross entropy loss (in data preprocessing stage, all sentences in a batch are padded to the same length, and the insufficient ones need to be supplemented), otherwise the culculation of gradient could be affected.

### Define training function
Parameters explanation:
    -- model.train(): enable trainig model, enable batch normalization and Dropout;
    -- clip_grad_norm: cutting gradients to prevent explosion. clip: gradient threshold;
    -- .view(): reducing the dimensions of output and trg for the loss culculation. The first column of output and trg will not be involved in calculations of loss in order to improve the accuracy becuase the beginning of each sentence of trg is the marker <sos>.

### Define testing function
    The difference between the evaluation phase and the training phase is that no parameters need to be updated.
    Parameters explanation:
    -- model.eval(): enable testing model, disable batch normalizaiton and dropout;
    -- torch.no_grad(): disable autograd engine (no back propagation calculation), the advantage is to reduce memory usage and speed up computing.
    -- teacher_forcing_ratio = 0: in testing phase, we need to turn off the teacher forcing to ensure that the model uses the predicted results as input for next step.

## Training model
--------

Parameters explanation:
    -- state_dict(): using the epoch parameter of the best verification loss as the final parameter of the model;
    -- math.exp(): using an average loss of a batch to calculate the values of perplexity(PPL)

## Verificate model
--------

Using the best parameters to test model in testing dataset.
    -- load_state_dict: loading the trained parameters.

## Testing model
--------

In order to test the performance of trained model we define a translate function and calculate the PPL at the same time. Inputing the source sentence manually which you need to translate, and test PPL value shows the performance of the trained model theoretically.



## Testing results
--------

### Theoretical criterion

Our hardware testing environment is based on Win10 OS, i7 9-Gen, 16G RAM and RTX2070 GPU. Training time and results has been list in Result.txt, and the testing result of model verification is as followed:

| Test Loss: 4.177 | Test PPL:  65.139 |

PPL(Preplexity) is a common criterion of language model, its basic idea is that it is better to give a higher probabiliby value to sentences in test sets. When the language model is trained and the sentence in the test set is normal, the higher probability in the test set, the better model is trained. From the definition and formular of PPL, we could know that a better language model should keep a lower value of PPL.

Our average test PPL is around 65 which is higher than the [baseline](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) of the paper in Reference, which could result from different target language and problems that have potential improvments.

### Practical criterion

A more straightforward way to test the trained model is checking the performance of model in practical applications(spelling check or machine translation). The function Translate.translate is to check the translation result of a single English sentence, which is treated as an intuitive approach in human habits.


## Reference
Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[J]. Advances in NIPS, 2014.
https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
https://stanfordnlp.github.io/stanfordnlp/
https://github.com/bentrevett/pytorch-seq2seq
https://pytorch-cn.readthedocs.io/zh/latest/
https://blog.csdn.net/leo_95/article/details/87708267#Field_7

