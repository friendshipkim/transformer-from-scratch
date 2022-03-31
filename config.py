import torch

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

# data hyperparameters (just for testing)
src_vocab_size = 10000
tgt_vocab_size = 15000
pad_idx = 1
src_seq_len = 100  # it differs by batch
tgt_seq_len = 120  # it differs by batch

# model architecture hyperparameters
n_layers = 6
d_model = emb_size = 512
ffn_hidden = d_model * 4
h = 8
p_drop = 0.1

# training hyperparameters
batch_size = 32
epochs = 15
lr = 0.0001
