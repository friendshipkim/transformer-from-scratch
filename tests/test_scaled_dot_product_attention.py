"""
Test code for scaled dot product attention

baseline torch.nn.functional._scaled_dot_product_attention
my: model.sublayer.MultiHeadAttention.calculate_attn

always the same without mask
"""

import torch
import torch.nn.functional as F
from model.sublayer.multihead_attention import MultiHeadAttention
import config as cfg
import os

seed = 3
scale = 100  # scale up input
# emb_file_path = "test_input/input_emb_nopad_32-10-512.pt"

def test_scaled_dot_product_attention():
    torch.manual_seed(seed)
    d_k = cfg.d_model // cfg.h

    # define input
    q = torch.rand((cfg.batch_size * cfg.h, cfg.src_seq_len, d_k)) * scale
    k = torch.rand((cfg.batch_size * cfg.h, cfg.tgt_seq_len, d_k)) * scale
    v = torch.rand((cfg.batch_size * cfg.h, cfg.tgt_seq_len, d_k)) * scale

    print("input size:", q.shape)

    # 1. without mask
    # baseline
    baseline_attn_output, baseline_attn_weights = F._scaled_dot_product_attention(q, k, v)
    # my
    my_attn = MultiHeadAttention(d_model=cfg.d_model, h=cfg.h, p_drop=0)
    my_attn_output, my_attn_weights = my_attn.calculate_attn(q, k, v)

    print("Attention outputs are the same?", torch.isclose(baseline_attn_output, my_attn_output).all().item())
    print("Attention weights are the same?", torch.isclose(baseline_attn_weights, my_attn_weights).all().item())


if __name__ == "__main__":
    test_scaled_dot_product_attention()
