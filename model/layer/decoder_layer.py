import typing
from torch import nn, Tensor

from model.sublayer.multihead_attention import MultiHeadAttention
from model.sublayer.pointwise_feedforward import PointwiseFeedForward



class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, p_drop: float):
        super(DecoderLayer, self).__init__()
        self.h = h

        self.self_attn = MultiHeadAttention(d_model, h, p_drop)
        self.dropout1 = nn.Dropout(p_drop)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, h, p_drop)
        self.dropout2 = nn.Dropout(p_drop)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = PointwiseFeedForward(d_model, ffn_hidden, p_drop)
        self.dropout3 = nn.Dropout(p_drop)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
            self, tgt_emb: Tensor, enc_output: Tensor, tgt_pad_mask: Tensor, tgt_autoregressive_mask: Tensor,
            memory_pad_mask: Tensor
    ) -> Tensor:
        """
        :param tgt_emb: torch.Tensor, Decoder input, shape: (batch_size, tgt_seq_len, d_model)
        :param enc_output: torch.Tensor, Encoder output, shape: (batch_size, src_seq_len, d_model)
        :param tgt_pad_mask: torch.Tensor, shape: (batch_size, tgt_seq_len)
        :param tgt_autoregressive_mask: torch.Tensor, shape: (tgt_seq_len, tgt_seq_len)
        :param memory_pad_mask: torch.Tensor, shape: (batch_size, src_seq_len)

        :return: torch.Tensor, shape: (batch_size, tgt_seq_len, d_model)
        """
        # create masks
        tgt_mask, cross_mask = self.create_dec_mask(tgt_pad_mask, tgt_autoregressive_mask, memory_pad_mask)

        # 1. self-attention
        self_attn_out, self_attn_score = self.self_attn(q=tgt_emb,
                                                        k=tgt_emb,
                                                        v=tgt_emb,
                                                        mask=tgt_mask)
        # hook mid outputs
        self.self_attn_out = self_attn_out
        self.self_attn_score = self_attn_score

        self_attn_out = self.dropout1(self_attn_out)
        self_attn_out = self.norm1(tgt_emb + self_attn_out)

        # 2. cross-attention
        cross_attn_out, cross_attn_score = self.cross_attn(q=self_attn_out,
                                                           k=enc_output,
                                                           v=enc_output,
                                                           mask=cross_mask)

        # hook mid outputs
        self.cross_attn_out = cross_attn_out
        self.cross_attn_score = cross_attn_score

        cross_attn_out = self.dropout2(cross_attn_out)
        cross_attn_out = self.norm2(self_attn_out + cross_attn_out)

        # 2. feed forward
        ffn_out = self.ffn(cross_attn_out)
        ffn_out = self.dropout3(ffn_out)
        dec_out = self.norm3(cross_attn_out + ffn_out)

        return dec_out

    def create_dec_mask(self, tgt_pad_mask: Tensor, tgt_autoregressive_mask: Tensor, memory_pad_mask: Tensor) \
            -> typing.Tuple[Tensor, Tensor]:
        # TODO maybe put this into multihead
        """
        create target padding + autoregressive mask
        create cross mask (= source (memory) padding mask)

        :param tgt_pad_mask: torch.Tensor, shape: (batch_size, tgt_seq_len)
        :param tgt_autoregressive_mask: shape: (tgt_seq_len, tgt_seq_len)
        :param memory_pad_mask: shape: (batch_size, src_seq_len)

        :return: tgt_mask: torch.Tensor, shape: (batch_size * h, tgt_seq_len, tgt_seq_len)
        :return: cross_mask: torch.Tensor, shape: (batch_size * h, tgt_seq_len, src_seq_len)
        """
        batch_size, tgt_seq_len = tgt_pad_mask.size()
        _, src_seq_len = memory_pad_mask.size()
        tgt_pad_mask = tgt_pad_mask.view(batch_size, 1, 1, tgt_seq_len). \
            expand(-1, self.h, tgt_seq_len, -1).reshape(batch_size * self.h, tgt_seq_len, tgt_seq_len)
        tgt_mask = tgt_autoregressive_mask.logical_and(tgt_pad_mask)

        cross_mask = memory_pad_mask.view(batch_size, 1, 1, src_seq_len). \
            expand(-1, self.h, tgt_seq_len, -1).reshape(batch_size * self.h, tgt_seq_len, src_seq_len)

        return tgt_mask, cross_mask
