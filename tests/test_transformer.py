import copy

import torch
from model.transformer import Transformer
from baseline_model.baseline_transformer import BaselineTransformer
import config as cfg
from test_utils import *
from torch import nn


def test_transformer():
    print("device:", cfg.device)
    torch.manual_seed(cfg.seed)
    torch.autograd.set_detect_anomaly(True)

    # baseline model
    baseline_model = BaselineTransformer(num_encoder_layers=cfg.n_layers,
                                         num_decoder_layers=cfg.n_layers,
                                         emb_size=cfg.d_model,
                                         nhead=cfg.h,
                                         src_vocab_size=cfg.src_vocab_size,
                                         tgt_vocab_size=cfg.tgt_vocab_size,
                                         dim_feedforward=cfg.ffn_hidden,
                                         dropout=cfg.p_drop,
                                         pad_idx=cfg.pad_idx,
                                         device=cfg.device)
    baseline_model.to(cfg.device)
    baseline_model.eval()

    # my model
    my_model = Transformer(n_layers=cfg.n_layers,
                           d_model=cfg.d_model,
                           h=cfg.h,
                           ffn_hidden=cfg.ffn_hidden,
                           p_drop=cfg.p_drop,
                           src_vocab_size=cfg.src_vocab_size,
                           tgt_vocab_size=cfg.tgt_vocab_size,
                           pad_idx=cfg.pad_idx,
                           device=cfg.device)
    my_model.to(cfg.device)
    my_model.eval()

    # 1. check # of params
    print("1. Check parameter counts")
    baseline_param = count_params(baseline_model)
    my_param = count_params(my_model)
    assert baseline_param == my_param, \
        "param number mismatch-> baseline: %d, mine: %d" % (baseline_param, my_param)
    # print("total # of params of baseline model:", baseline_param)
    # print("total # of params of my model:", my_param)
    print("Params count checked!")

    # 2. check output shape
    print("\n2. Check model output shape")
    enc_input_nopad = generate_input_nopad(seed=cfg.seed,
                                           pad_idx=cfg.pad_idx,
                                           vocab_size=cfg.src_vocab_size,
                                           seq_len=cfg.src_seq_len,
                                           batch_size=cfg.batch_size).to(cfg.device)
    dec_input_nopad = generate_input_nopad(seed=cfg.seed,
                                           pad_idx=cfg.pad_idx,
                                           vocab_size=cfg.tgt_vocab_size,
                                           seq_len=cfg.tgt_seq_len,
                                           batch_size=cfg.batch_size).to(cfg.device)
    _, _, _, _, my_out_test = my_model(enc_input_nopad, dec_input_nopad)
    _, _, _, _, baseline_out_test = baseline_model(enc_input_nopad, dec_input_nopad)

    assert baseline_out_test.shape == my_out_test.shape, \
        "output shape mismatch-> baseline: %d, mine: %d" % (baseline_out_test.shape, my_out_test.shape)
    # print("baseline model output shape:", baseline_out_test.shape)
    # print("my model output shape:", my_out_test.shape)
    print("Model output shape checked!")

    # 3. copy state dict
    print("\n3. Copy model parameters (baseline -> my) and check model weights shape")
    copied_my_sd = copy_transformer_dict(src=baseline_model.state_dict(),
                                         tgt=copy.deepcopy(my_model.state_dict()),
                                         n_layers=cfg.n_layers)
    my_model.load_state_dict(copied_my_sd)

    # test if baseline and my_copied weights are the same
    assert torch.isclose(my_model.state_dict()['src_tok_emb.embedding.weight'], \
                         baseline_model.state_dict()['src_tok_emb.embedding.weight']).all()
    print("Model weights have the same shape")

    print("\n4-1. Feed the input without padding")
    baseline_src_emb, baseline_tgt_emb, baseline_enc_out, baseline_dec_out, baseline_out = baseline_model(
        enc_input_nopad,
        dec_input_nopad)
    my_src_emb, my_enc_out, my_tgt_emb, my_dec_out, my_out = my_model(enc_input_nopad,
                                                                      dec_input_nopad)

    assert torch.isclose(my_out, baseline_out).all(), "(without padding) output of baseline and my model are different"
    print("(without padding) baseline and my outputs are the same")

    print("\n4-2. Feed the input with padding")
    # new input with padding
    enc_input_pad = generate_input_pad(seed=cfg.seed,
                                       pad_idx=cfg.pad_idx,
                                       vocab_size=cfg.src_vocab_size,
                                       seq_len=cfg.src_seq_len,
                                       batch_size=cfg.batch_size).to(cfg.device)
    dec_input_pad = generate_input_pad(seed=cfg.seed,
                                       pad_idx=cfg.pad_idx,
                                       vocab_size=cfg.tgt_vocab_size,
                                       seq_len=cfg.tgt_seq_len,
                                       batch_size=cfg.batch_size).to(cfg.device)

    # compare the output
    baseline_src_emb, baseline_tgt_emb, baseline_enc_out, baseline_dec_out, baseline_out = baseline_model(enc_input_pad,
                                                                                                          dec_input_pad)
    my_src_emb, my_enc_out, my_tgt_emb, my_dec_out, my_out = my_model(enc_input_pad,
                                                                      dec_input_pad)
    assert torch.isclose(baseline_out, my_out).all(), "(with padding) output of baseline and my model are different"
    print("(with padding) baseline and my outputs are the same")

    print("\n5. Check gradients")
    # targets - decoder input shifted by 1
    bos = torch.ones(cfg.batch_size, dtype=torch.long) * 2
    bos = bos.unsqueeze(-1).to(cfg.device)
    shifted_target = dec_input_pad[:, 1:]
    targets = torch.cat((bos, shifted_target), dim=1).flatten()

    # loss
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_idx)

    # baseline
    baseline_loss = criterion(baseline_out.view(-1, cfg.tgt_vocab_size), targets)
    baseline_loss.backward()

    # my
    my_loss = criterion(my_out.view(-1, cfg.tgt_vocab_size), targets)
    my_loss.backward()

    # check if gradients are the same
    check_transformer_grads(baseline_model, my_model)

if __name__ == "__main__":
    test_transformer()
