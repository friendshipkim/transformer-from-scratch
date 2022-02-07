from torch import nn

from model.layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, h, ffn_hidden, p_drop):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, ffn_hidden, p_drop)
                                     for _ in range(n_layers)])

    def forward(self, x, enc_output, tgt_mask, cross_mask):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, cross_mask)
        return x
