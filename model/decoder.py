from torch import nn

from model.layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, n_layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
