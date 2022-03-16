from torch import nn
from torch import Tensor

import torch

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, p_drop: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)

        self.ffn = PointwiseFeedForward(d_model, ffn_hidden, p_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)

        self.attn_out_weights = None

    def forward(self, x: Tensor, enc_mask: Tensor = None) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :param enc_mask: torch.Tensor, shape (batch_size, h, q_max_seq_len, k_max_seq_len)
        :return: torch.Tensor, shape (batch_size, max_seq_len, d_model)
        """

        # 1. attention
        attn_out_dict = self.self_attn(q=x, k=x, v=x, mask=enc_mask)
        attn_out = attn_out_dict["output"]
        attn_out_weights = attn_out_dict["attn_score"]
        out = self.dropout1(attn_out)
        out = self.norm1(x + out)

        self.attn_out_weights = attn_out_weights

        # torch.save(x, f'/home/wk247/workspace/transformer-from-scratch/tensors/my.encoder.{i}.selfattn.norm.pt')
        # print(f'saving mine - {i}th layer after selfattn')

        # 2. feed forward
        cp_out = out
        out = self.ffn(out)
        out = self.dropout2(out)
        out = self.norm2(out + cp_out)

        # torch.save(x, f'/home/wk247/workspace/transformer-from-scratch/tensors/my.encoder.{i}.ffn.norm.pt')
        # print(f'saving mine - {i}th layer after ffn')

        return out
