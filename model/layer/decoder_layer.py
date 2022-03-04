from torch import nn, Tensor

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, p_drop: float):
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

    def forward(self, x: Tensor, enc_output: Tensor, dec_mask: Tensor, cross_mask: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, Decoder input, shape: (batch_size, max_seq_len, d_model)
        :param enc_output: torch.Tensor, Encoder output, shape: (batch_size, max_seq_len, d_model)
        :param dec_mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)
        :param cross_mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)

        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        # 1. self-attention
        x_cp = x
        x = self.attn_self(q=x, k=x, v=x, mask=dec_mask)["output"]
        x = self.norm1(x + x_cp)
        x = self.dropout1(x)

        # 2. cross-attention
        x_cp = x
        x = self.attn_cross(q=x, k=enc_output, v=enc_output, mask=cross_mask)["output"]
        x = self.norm2(x + x_cp)
        x = self.dropout2(x)

        # 2. feed forward
        x_cp = x
        x = self.ffn(x)
        x = self.norm3(x + x_cp)
        x = self.dropout3(x)

        return x
