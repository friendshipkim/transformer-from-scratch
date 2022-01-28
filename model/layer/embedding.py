import torch
import math
from torch import nn


class Embedding(nn.Module):
    def __init__(self, d_embed, vocab):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(len(vocab), d_embed)
        self.vocab = vocab
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_seq_len, d_embed)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.encoding = encoding

    def forward(self, x):
        x = x + torch.Variable(self.encoding[:, :x.size(1)], requires_grad=False)
        return x
