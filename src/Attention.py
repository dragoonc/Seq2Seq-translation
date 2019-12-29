import torch.nn as nn
import torch
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention,self).__init__()
        self.enc_hid_dim=enc_hid_dim
        self.dec_hid_dim=dec_hid_dim
        self.attn=nn.Linear(enc_hid_dim*2+dec_hid_dim,dec_hid_dim)
        ### self.attn = nn.Linear(enc_hid_dim+dec_hid_dim, dec_hid_dim)
        self.v=nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        batch_size=encoder_outputs.shape[1]
        src_len=encoder_outputs.shape[0]
        #重复操作，让隐藏状态的第二个维度和encoder相同
        hidden=hidden.unsqueeze(1).repeat(1,src_len,1)
        #该函数按指定的向量来重新排列一个数组，在这里是调整encoder输出的维度顺序，在后面能够进行比较
        encoder_outputs=encoder_outputs.permute(1,0,2)
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        #开始计算hidden和encoder_outputs之间的匹配值
    #    energy=torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        energy_cat = torch.cat((hidden, encoder_outputs), dim=2)      ######## for testing
        energy_attn = self.attn(energy_cat)                           ######## for testing
        energy = torch.tanh(energy_attn)                              ######## for testing 
        #energy = [batch size, src sent len, dec hid dim]
        #调整energy的排序
        energy=energy.permute(0,2,1)
        #energy = [batch size, dec hid dim, src sent len]
        
        #v = [dec hid dim]
        v=self.v.repeat(batch_size,1).unsqueeze(1)
        #v = [batch_size, 1, dec hid dim] 注意这个bmm的作用，对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操
        attention=torch.bmm(v,energy).squeeze(1)
        #attention=[batch_size, src_len]
        return F.softmax(attention, dim=1)
