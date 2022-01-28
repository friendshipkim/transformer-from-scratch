from torch import nn

from model.layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
