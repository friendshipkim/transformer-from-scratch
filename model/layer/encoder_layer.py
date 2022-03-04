from torch import nn
from torch import Tensor

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, p_drop: int):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)

        self.ffn = PointwiseFeedForward(d_model, ffn_hidden)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, x: Tensor, enc_mask: Tensor = None) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :param enc_mask: torch.Tensor, shape (batch_size, h, q_max_seq_len, k_max_seq_len)
        :return: torch.Tensor, shape (batch_size, max_seq_len, d_model)
        """

        # 1. attention
        x_out_dict = self.attn(q=x, k=x, v=x, mask=enc_mask)  # TODO: encoder mask (future mask)
        x_out = x_out_dict["output"]

        x_out = self.norm1(x + x_out)
        x_out = self.dropout1(x_out)

        # 2. feed forward
        cp_x_out = x_out
        x_out = self.ffn(x_out)
        x_out = self.norm2(x_out + cp_x_out)
        x_out = self.dropout2(x_out)

        return x_out
