import torch
from model.transformer import Transformer
from config import *
import numpy as np


def test_transformer():
    # input - output of embedding layer
    enc_input = torch.randint(vocab_size, (batch_size, max_seq_len))
    dec_input = torch.randint(vocab_size, (batch_size, max_seq_len))

    model = Transformer(n_layers=n_layers,
                        d_model=d_model,
                        h=h,
                        ffn_hidden=ffn_hidden,
                        p_drop=p_drop,
                        vocab_size=vocab_size,
                        max_seq_len=max_seq_len,
                        pad_idx=0, # TODO
                        device=device)  # TODO

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total # of params:", params)

    # transformer output
    out = model(enc_input, dec_input)
    print("output shape:", out.shape)

    return out


if __name__ == "__main__":
    test_transformer()
