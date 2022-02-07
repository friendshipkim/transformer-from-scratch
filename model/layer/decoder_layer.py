from torch import nn

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, ffn_hidden, p_drop):
        super(DecoderLayer, self).__init__()
        self.attn_self = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)

        self.attn_cross = MultiHeadAttention(d_model, h)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)

        self.ffn = PointwiseFeedForward(d_model, ffn_hidden)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p_drop)

    def forward(self, x, enc_output, dec_mask, cross_mask):
        # 1. self-attention
        x_cp = x
        x = self.attn_self(q=x, k=x, v=x, mask=dec_mask)
        x = self.norm1(x + x_cp)
        x = self.dropout1(x)

        # 2. cross-attention
        x_cp = x
        x = self.attn_cross(q=x, k=enc_output, v=enc_output, mask=cross_mask)
        x = self.norm2(x + x_cp)
        x = self.dropout2(x)

        # 2. feed forward
        x_cp = x
        x = self.ffn(x)
        x = self.norm3(x + x_cp)
        x = self.dropout3(x)

        return x
