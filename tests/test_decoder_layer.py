import torch
from model.layer.decoder_layer import DecoderLayer
from config import *


def test_decoder_layer():
    # input
    x = torch.rand(
        (batch_size, max_seq_len, d_model)
    )  # output of decoder embedding layer
    enc_output = torch.rand((batch_size, max_seq_len, d_model))

    # decoder_mask, cross_mask
    q_max_seq_len = k_max_seq_len = max_seq_len
    dec_mask = torch.randint(
        2, (batch_size, h, q_max_seq_len, k_max_seq_len)
    )  # TODO: make dec_mask
    cross_mask = torch.randint(
        2, (batch_size, h, q_max_seq_len, k_max_seq_len)
    )  # TODO: make cross_mask

    # decoder layer
    decoder_layer = DecoderLayer(
        d_model=d_model, h=h, ffn_hidden=ffn_hidden, p_drop=p_drop
    )

    # decoder layer output
    out = decoder_layer(x, enc_output, dec_mask, cross_mask)

    return out


if __name__ == "__main__":
    test_decoder_layer()
