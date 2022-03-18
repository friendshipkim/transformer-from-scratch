"""
Case 2. Test code for padded input

a) float32
embedding output - always the same
encoder output - differs sometimes (for small numbers?),
               - atol = 1e-6 or 1e-7, precision error gets larger when d_model or batch_size is larger

b) float64 = double
embedding output, encoder_output - always the same
"""

import copy
import os

import torch

from model.sublayer.positional_encoding import PositionalEncoding as my_PE
from model.sublayer.token_embedding import TokenEmbedding as my_TE
from model.baseline_transformer import PositionalEncoding as baseline_PE
from model.baseline_transformer import TokenEmbedding as baseline_TE

from torch.nn import TransformerEncoderLayer as BaselineEncLayer
from model.layer.encoder_layer import EncoderLayer as MyEncLayer

import config as cfg
from layer_mapping import *

# global variables
load_input_flag = False  # True if to load stored input tensor
save_input_flag = False  # True if to save generated random input tensor
input_file_path = 'test_input/input_pad_2by10.pt'  # file path to the stored input tensor
seed = 130
double_flag = False  # True if to test with double dtype


def generate_input_pad(seed, save_flag=False, file_path=None):
    torch.manual_seed(seed)
    valied_token_counts = torch.randint(1, cfg.src_seq_len,
                                        (cfg.batch_size,))  # number of valid tokens (not <pad>) in each input

    input_pad = torch.empty(size=(cfg.batch_size, cfg.src_seq_len), dtype=torch.long)
    for i in range(cfg.batch_size):
        valid_length = valied_token_counts[i]
        pad_count = cfg.src_seq_len - valid_length

        valid = torch.randint(cfg.pad_idx + 1, cfg.src_vocab_size, (valid_length,))  # make 1d tensor (one input)
        padding = torch.LongTensor([cfg.pad_idx] * pad_count)
        sample = torch.cat((valid, padding), dim=0)
        input_pad[i] = sample

    print("input_pad shape:", input_pad.shape)
    print("input_pad:", input_pad)

    if save_flag:
        current_path = os.path.dirname(os.path.realpath(__file__))
        torch.save(input_pad, os.path.join(current_path, file_path))
        print("input tensor saved")

    return input_pad


def load_input_pad(file_path):
    current_path = os.path.dirname(os.path.realpath(__file__))
    input_pad = torch.load(os.path.join(current_path, file_path))
    print("input tensor loaded")
    print("input_pad shape:", input_pad.shape)
    print("input_pad:", input_pad)
    return input_pad


def same_size(x, y):
    # check if x and y has the same size
    return x.shape == y.shape


def copy_te_dict(src, tgt):
    copy_count = 0
    te_mapping = {"embedding.weight": "embedding.weight"}
    for src_layer_name, tgt_layer_name in te_mapping.items():
        if same_size(src[src_layer_name], tgt[tgt_layer_name]):
            tgt[tgt_layer_name] = src[src_layer_name]
            copy_count += 1
        else:
            assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
    assert copy_count == len(tgt), "Token embedding is not copied correctly"
    print("Token embedding copied: baseline -> my")
    return tgt


def copy_pe_dict(src, tgt):
    copy_count = 0
    pe_mapping = {"pos_embedding": "encoding"}
    for src_layer_name, tgt_layer_name in pe_mapping.items():
        if same_size(src[src_layer_name], tgt[tgt_layer_name]):
            tgt[tgt_layer_name] = src[src_layer_name]
            copy_count += 1
        else:
            assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
    assert copy_count == len(tgt), "Positional encoding is not copied correctly"
    print("Positional encoding copied: baseline -> my")
    return tgt


def copy_enc_layer_dict(src, tgt):
    copy_count = 0
    # postfix dict
    postfix_mapping = encoder_postfix_mapping

    for src_postfix, tgt_postfix in postfix_mapping.items():
        if type(tgt_postfix) == list:  # a baseline layer maps to multiple my layers
            src_layer_name = src_postfix
            tgt_layer_names = tgt_postfix

            src_concat_weight = src[src_layer_name]  # shape: (d_model * 3, d_model) for weights
            src_weights = torch.split(src_concat_weight, src_concat_weight.size(0) // 3, dim=0)

            for src_weight, tgt_layer_name in zip(src_weights, tgt_layer_names):
                if same_size(tgt[tgt_layer_name], src_weight):
                    tgt[tgt_layer_name] = src_weight
                    copy_count += 1
                else:
                    assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
        else:
            src_layer_name = src_postfix
            tgt_layer_name = tgt_postfix

            if same_size(src[src_layer_name], tgt[tgt_layer_name]):
                tgt[tgt_layer_name] = src[src_layer_name]
                copy_count += 1
            else:
                assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
    assert len(tgt) == copy_count, "Encoder layer is not copied correctly"
    print("Encoder layer copied: baseline -> my")
    return tgt


def create_baseline_src_masks(src, device, pad_idx):
    src_seq_len = src.size(1)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    src_padding_mask = (src == pad_idx)

    return src_mask, src_padding_mask


def create_my_src_masks(x, pad_idx):  # creates padding mask
    """
    Create a mask to hide padding

    :param x: torch.Tensor, shape: (batch_size, seq_len)
    :param pad_idx: int, pad token index

    :return: torch.Tensor, shape: (batch_size, 1, seq_len)
    """
    return (x != pad_idx).unsqueeze(-2)


def create_my_src_masks_deprecated(q, k, device, pad_idx):
    """
    Create a mask to hide padding

    :param q: torch.Tensor, shape: (batch_size, q_seq_len)
    :param k: torch.Tensor, shape: (batch_size, k_seq_len)
    :param pad_idx: int, pad token index

    :return: torch.Tensor, shape: (batch_size, 1, q_seq_len, k_seq_len)
    """
    len_q, len_k = q.size(1), k.size(1)

    # batch_size x 1 x 1 x len_k
    k = k.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    # batch_size x 1 x len_q x len_k
    k = k.repeat(1, 1, len_q, 1)

    # batch_size x 1 x len_q x 1
    q = q.ne(pad_idx).unsqueeze(1).unsqueeze(3)
    # batch_size x 1 x len_q x len_k
    q = q.repeat(1, 1, 1, len_k)

    mask = k & q
    return mask.to(device)  # shape: (batch_size, 1, q_seq_len, k_seq_len) it can be also broadcasted


def test_encoder_pad():
    # manual seed
    torch.manual_seed(seed)

    # baseline model
    # 1. embedding
    baseline_src_te = baseline_TE(vocab_size=cfg.src_vocab_size, emb_size=cfg.d_model)
    baseline_pe = baseline_PE(emb_size=cfg.emb_size, dropout=cfg.p_drop)

    baseline_src_te.to(cfg.device)
    baseline_pe.to(cfg.device)

    baseline_src_te.eval()
    baseline_pe.eval()

    # 2. encoder layer
    baseline_enc_layer = BaselineEncLayer(d_model=cfg.d_model,
                                          nhead=cfg.h,
                                          dim_feedforward=cfg.ffn_hidden,
                                          dropout=cfg.p_drop,
                                          batch_first=True,
                                          device=cfg.device)
    baseline_enc_layer.to(cfg.device)
    baseline_enc_layer.eval()

    # my model
    # 1. embedding
    my_src_te = my_TE(emb_size=cfg.d_model, vocab_size=cfg.src_vocab_size)
    my_pe = my_PE(emb_size=cfg.d_model, p_drop=cfg.p_drop)

    my_src_te.to(cfg.device)
    my_pe.to(cfg.device)

    my_src_te.eval()
    my_pe.eval()

    # 2. encoder layer
    my_enc_layer = MyEncLayer(d_model=cfg.d_model,
                              h=cfg.h,
                              ffn_hidden=cfg.ffn_hidden,
                              p_drop=cfg.p_drop)
    my_enc_layer.to(cfg.device)
    my_enc_layer.eval()

    # copy state_dict
    # 1-1. token embedding
    copied_my_src_te_sd = copy_te_dict(src=baseline_src_te.state_dict(),
                                       tgt=copy.deepcopy(my_src_te.state_dict()))
    my_src_te.load_state_dict(copied_my_src_te_sd)
    assert torch.isclose(baseline_src_te.state_dict()["embedding.weight"],
                         my_src_te.state_dict()["embedding.weight"]).all().item()

    # 1-2. positional encoding
    copied_my_pe_sd = copy_pe_dict(src=baseline_pe.state_dict(),
                                   tgt=copy.deepcopy(my_pe.state_dict()))
    my_pe.load_state_dict(copied_my_pe_sd)
    assert torch.isclose(baseline_pe.state_dict()["pos_embedding"], my_pe.state_dict()["encoding"]).all().item()

    # 2. encoder layer
    copy_my_enc_sd = copy_enc_layer_dict(src=baseline_enc_layer.state_dict(),
                                         tgt=copy.deepcopy(my_enc_layer.state_dict()))
    my_enc_layer.load_state_dict(copy_my_enc_sd)

    # convert models to double
    if double_flag:
        module_list = [my_src_te, my_pe, my_enc_layer,
                       baseline_src_te, baseline_pe, baseline_enc_layer]
        for module in module_list:
            module.double()

    # feed an input
    print("=" * 30, "feed the input with padding", "=" * 30)
    if load_input_flag:  # load saved input
        input_pad = load_input_pad(input_file_path).to(cfg.device)
    else:
        input_pad = generate_input_pad(seed).to(cfg.device)  # generate rand input

    # create masks
    # baseline_src_mask shape: (src_seq_len, src_seq_len)
    # baseline_src_padding_mask shape: (batch_size, src_seq_len)
    # my_src_padding_mask shape: (batch_size, 1, src_seq_len)
    baseline_src_mask, baseline_src_padding_mask = create_baseline_src_masks(src=input_pad, device=cfg.device, pad_idx=cfg.pad_idx)
    my_src_padding_mask = create_my_src_masks(x=input_pad, pad_idx=cfg.pad_idx)

    breakpoint()
    # embedding output, shape: (batch_size,src_seq_len, d_model)
    baseline_src_emb = baseline_pe(baseline_src_te(input_pad))
    my_src_emb = my_pe(my_src_te(input_pad))
    emb_flag = torch.isclose(baseline_src_emb, my_src_emb).all().item()
    print("Embedding outputs are the same?:", emb_flag)

    # encoder layer output, shape: (batch_size, src_seq_len, d_model)
    my_enc_out = my_enc_layer(x=my_src_emb,
                              enc_mask=my_src_padding_mask)
    baseline_enc_out = baseline_enc_layer(src=baseline_src_emb,
                                          src_mask=baseline_src_mask,
                                          src_key_padding_mask=baseline_src_padding_mask)

    # compare encoder output
    unmatch_indices = (torch.isclose(baseline_enc_out, my_enc_out) == False).nonzero(as_tuple=True)
    baseline_unmatch = baseline_enc_out[unmatch_indices]
    my_unmatch = my_enc_out[unmatch_indices]

    atol = 1e-07
    print("unmatched count:", len(baseline_unmatch))
    print("compare with low precision:", torch.isclose(baseline_unmatch, my_unmatch, atol=atol))

    breakpoint()


if __name__ == "__main__":
    # generate_input_pad(seed, save_flag=save_input_flag, file_path=input_file_path)
    test_encoder_pad()
