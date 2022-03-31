from torch import nn, Tensor

from model.layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self, n_layers: int, d_model: int, h: int, ffn_hidden: int, p_drop: float
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, ffn_hidden, p_drop) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, tgt_emb: Tensor, enc_output: Tensor, tgt_pad_mask: Tensor, tgt_autoregressive_mask: Tensor, memory_pad_mask: Tensor
    ) -> Tensor:
        """
        :param tgt_emb: torch.Tensor, Decoder input, shape: (batch_size, max_seq_len, d_model)
        :param enc_output: torch.Tensor, Encoder output, shape: (batch_size, src_seq_len, d_model)
        :param tgt_pad_mask: torch.Tensor, shape: (batch_size, tgt_seq_len)
        :param tgt_autoregressive_mask: torch.Tensor, shape: (tgt_seq_len, tgt_seq_len)
        :param memory_pad_mask: shape: (batch_size, src_seq_len)

        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        out = tgt_emb
        for layer in self.layers:
            out = layer(out, enc_output, tgt_pad_mask, tgt_autoregressive_mask, memory_pad_mask)
        out = self.norm(out)
        return out
