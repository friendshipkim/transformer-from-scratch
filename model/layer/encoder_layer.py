from torch import nn
from torch import Tensor

import torch

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, p_drop: float):
        super(EncoderLayer, self).__init__()
        self.h = h

        self.self_attn = MultiHeadAttention(d_model, h, p_drop)
        self.dropout1 = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = PointwiseFeedForward(d_model, ffn_hidden, p_drop)
        self.dropout2 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src_emb: Tensor, src_pad_mask: Tensor = None) -> Tensor:
        """
        :param src_emb: torch.Tensor, shape: (batch_size, src_seq_len, d_model)
        :param src_pad_mask: torch.Tensor, shape (batch_size, src_seq_len)
        :return: torch.Tensor, shape (batch_size, src_seq_len, d_model)
        """

        # 1. attention
        attn_out, attn_score = self.self_attn(q=src_emb, k=src_emb, v=src_emb,
                                              mask=None if src_pad_mask == None
                                                        else self.create_enc_mask(pad_mask=src_pad_mask))
        # hook mid outputs
        self.attn_score = attn_score
        self.attn_out = attn_out

        attn_out = self.dropout1(attn_out)
        attn_out = self.norm1(src_emb + attn_out)

        # 2. feed forward
        ffn_out = self.ffn(attn_out)
        ffn_out = self.dropout2(ffn_out)
        enc_out = self.norm2(attn_out + ffn_out)

        return enc_out

    def create_enc_mask(self, pad_mask: Tensor):
        # TODO maybe put this into multihead
        """
        transform source padding mask into proper shape

        :param pad_mask: torch.Tensor, shape: (batch_size, src_seq_len)
        :return: enc_mask: torch.Tensor, shape: (batch_size * h, src_seq_len, src_seq_len)
        """
        batch_size, src_seq_len = pad_mask.size()
        enc_mask = pad_mask.view(batch_size, 1, 1, src_seq_len). \
            expand(-1, self.h, src_seq_len, -1).reshape(batch_size * self.h, src_seq_len, src_seq_len)
        return enc_mask
