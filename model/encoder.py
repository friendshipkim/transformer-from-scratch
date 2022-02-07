from torch import nn

from model.layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, h, ffn_hidden, p_drop):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, ffn_hidden, p_drop)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
