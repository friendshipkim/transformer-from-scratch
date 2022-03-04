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

    def forward(
        self, x: Tensor, enc_output: Tensor, dec_mask: Tensor, cross_mask: Tensor
    ) -> Tensor:
        """
        :param x: torch.Tensor, Decoder input, shape: (batch_size, max_seq_len, d_model)
        :param enc_output: torch.Tensor, Encoder output, shape: (batch_size, max_seq_len, d_model)
        :param dec_mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)
        :param cross_mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)

        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, enc_output, dec_mask, cross_mask)
        return x
