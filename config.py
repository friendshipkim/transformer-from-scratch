import torch

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data hyperparameters
vocab_size = 10000

# model architecture hyperparameters
n_layers = 6
d_model = 256
ffn_hidden = 2048
h = 8
p_drop = 0.1
max_seq_len = 100  # TODO: check paper

# training hyperparameters
batch_size = 8
