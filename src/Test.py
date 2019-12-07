"""
@file: Test.py
@time: 2019/11/30
@desc: A time function is set to calculate the time consumed by each epoch round; the output word is used as the prediction result through the probability distribution of all words
"""

import Paras
import torch
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def greedy_search(src, model, device, TRG):
    eos_tok = TRG.vocab.stoi['<eos>']
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    #batch_size = TRG.shape[1]
    max_len = Paras.MAX_LEN
    #trg_vocab_size = model.decoder.output_dim

    # tensor to store decoder outputs
    #outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(model.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    hidden, cell = model.encoder(src)

    # first input to the decoder is the <sos> tokens
    #input = torch.ones(1, 1 , 1) * TRG.vocab.stoi['<sos>']
    #lengths=torch.tensor([len(TRG.vocab.stoi['<sos>'])])
    #a=TRG.vocab.stoi['<sos>']
    input=torch.LongTensor([TRG.vocab.stoi['<sos>']])
    print("输入")
    print(input.shape)
    for t in range(1, max_len):
        prediction, hidden, cell = model.decoder(input, hidden, cell)
        print("prediction")
        print(prediction)       #得到所有单词的概率分布
        #prediction = [batch size, output dim]
        #output = F.softmax(prediction, dim=-2)  #有些dim等于-1，有些等于1？
        output = prediction.max(1)[1]        #从所有单词的概率分布中选择最大的概率，作为输出单词
        #outputs[t] = output
        print("output")
        output = output.long()       # 将floatTensor转成longTensor，方便下面的cat操作
        print(output)
        all_tokens = torch.cat((all_tokens, output), dim=0)
        print(all_tokens)
        # teacher_force = random.random() < teacher_forcing_ratio
        # top1 = output.max(1)[1]    #这个是最大的输出概率吗？
        input = output   # 如何把torch.Size([1,5618])变成torch.Size([1])


        """topv,topi=output.data.topk(1)
        topi=topi.view(-1)
        decoded_batch[:,t]=topi"""
    print("all_tokens")
    print(all_tokens)
    # print(all_tokens.numpy().tolist().count(eos_tok))
    length = (all_tokens == eos_tok).nonzero()[0]

    return ' '.join([TRG.vocab.itos[tok] for tok in all_tokens[1:length]])
