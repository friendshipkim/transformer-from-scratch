"""
deprecated
"""

import torch
from model.sublayer.multihead_attention import MultiHeadAttention


def test_attention():
    # set params
    d_model = 256
    h = 4
    batch_size = 16
    p_drop = 0
    # use different q_max_seq_length and k_max_seq_len for test
    q_max_seq_len = 10
    k_max_seq_len = 20
    v_max_seq_len = k_max_seq_len

    # inputs
    q = torch.rand((batch_size, q_max_seq_len, d_model))
    k = torch.rand((batch_size, k_max_seq_len, d_model))
    v = torch.rand((batch_size, v_max_seq_len, d_model))

    # mask if mask == 0
    # mask = torch.randint(2, (q_max_seq_len, k_max_seq_len)) # broadcast batch_size * h
    mask = torch.randint(2, (batch_size, 1, q_max_seq_len, k_max_seq_len))  # broadcast h
    # mask = torch.randint(2, (batch_size, 1, q_max_seq_len))  # broadcast - this doesn't work
    # mask = torch.randint(2, (batch_size, h, q_max_seq_len, k_max_seq_len)) - original

    # attention output
    attn = MultiHeadAttention(d_model, h, p_drop)
    out_dict = attn(q, k, v, mask=mask)
    out = out_dict["output"]
    attn_score = out_dict["attn_score"]
    # print(attn_score)

    # check if mask is correctly applied
    assert (mask == 0).sum() * h == (attn_score == 0).sum()
    # check if the shape is right
    assert out.size() == (batch_size, q_max_seq_len, d_model)

    return out


if __name__ == "__main__":
    test_attention()
