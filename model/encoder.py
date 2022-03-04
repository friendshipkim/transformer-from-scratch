from torch import nn, Tensor
from model.layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, h: int, ffn_hidden: int, p_drop: float):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, ffn_hidden, p_drop)
                                     for _ in range(n_layers)])

    def forward(self, x: Tensor, enc_mask: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, Encoder input, shape: (batch_size, max_seq_len, d_model)
        :param enc_mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)

        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, enc_mask)
        return x
