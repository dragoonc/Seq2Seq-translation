"""
@file: Paras.py
@time: 2019/12/24
@desc: Define parameters, network structure and training related parameters used to create SRC and TRG field objects
"""
# tokens for head and rear
BOS_WORD = "<sos>"
EOS_WORD = "<eos>"
BLANK_WORD = "<blank>"
MAX_LEN = 30


ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
