from torch import nn

from model.layer.embedding import Embedding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embedding = Embedding()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, z):
        """
        x - encoder input, z - decoder input
        """
        x = self.encoder(x)  # context
        x = self.decoder(x, z)
        return x