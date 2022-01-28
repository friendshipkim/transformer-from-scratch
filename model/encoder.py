from torch import nn
import copy


class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
