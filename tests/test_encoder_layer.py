import torch
from model.layer.encoder_layer import EncoderLayer
from config import *


def test_encoder_layer():
    # input - output of embedding layer
    x = torch.rand((batch_size, max_seq_len, d_model))

    encoder_layer = EncoderLayer(
        d_model=d_model, h=h, ffn_hidden=ffn_hidden, p_drop=p_drop
    )

    # encoder mask
    q_max_seq_len = k_max_seq_len = max_seq_len
    mask = torch.randint(2, (batch_size, h, q_max_seq_len, k_max_seq_len))

    # encoder layer output
    out = encoder_layer(x, mask)  # TODO: check with enc_mask

    return out


if __name__ == "__main__":
    test_encoder_layer()
