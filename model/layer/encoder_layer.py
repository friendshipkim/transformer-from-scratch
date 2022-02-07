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

    def forward(self, x, mask=None):
        # 1. attention
        x_out_dict = self.attention(q=x, k=x, v=x, mask=mask)
        x_out = x_out_dict["output"]

        x_out = self.norm1(x + x_out)
        x_out = self.dropout1(x_out)

        # 2. feed forward
        x_out2 = self.ffn(x_out)
        x_out2 = self.norm2(x_out + x_out2)
        x_out2 = self.dropout2(x_out2)

        return x_out2
