import copy

import torch
from torch import nn

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, ffn_hidden, p_drop):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)

        self.ffn = PointwiseFeedForward(d_model, ffn_hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)

    def forward(self, x):
        # 1. attention
        x_cp = copy.deepcopy(x)
        x = self.attention(q=x, k=x, v=x, mask=None)
        x = self.norm1(x + x_cp)
        x = self.dropout1(x)

        # 2. feed forward
        x_cp = copy.deepcopy(x)
        x = self.ffn(x)
        x = self.norm2(x + x_cp)
        x = self.dropout2(x)

        return x
