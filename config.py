import torch

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

# data hyperparameters
# src_vocab_size = 10000
# tgt_vocab_size = 15000
# pad_idx = 1

# model architecture hyperparameters
n_layers = 6
d_model = emb_size = 512
ffn_hidden = d_model * 4
h = 8
p_drop = 0.1
# src_seq_len = 100  # it differs by batch
# tgt_seq_len = 120  # it differs by batch

# training hyperparameters
batch_size = 32
eval_batch_size = 16  # TODO: check if it is necessary
