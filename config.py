import torch

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data hyperparameters
src_vocab_size = 10
tgt_vocab_size = 15
pad_idx = 1  # TODO

# model architecture hyperparameters
n_layers = 6
d_model = emb_size = 32
ffn_hidden = d_model * 4
h = 8
p_drop = 0.1
src_seq_len = 10
tgt_seq_len = 12  # TODO: it differs by batch

# training hyperparameters
batch_size = 2
eval_batch_size = 16  # TODO: check if it is necessary
