import copy

import torch
from model.transformer import Transformer
from model.baseline_transformer import BaselineTransformer
import config as cfg
import numpy as np

from layer_mapping import *


# import fairseq


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count


def same_size(x, y):
    # check if x and y has the same size
    return x.shape == y.shape


def check_state_dict(baseline_sd, my_sd):
    # check for embedding
    for baseline_layer_name, my_layer_name in embedding_mapping.items():
        baseline_layer = baseline_sd.pop(baseline_layer_name)
        my_layer = my_sd.pop(my_layer_name)
        assert baseline_layer.shape == my_layer.shape
    print("embedding test pass")

    # check for transformer blocks
    for enc_dec in ["encoder", "decoder"]:
        # for every layer
        for layer in range(cfg.n_layers):
            # prefix
            baseline_prefix = f"transformer.{enc_dec}.layers.{layer}"
            my_prefix = f"{enc_dec}.layers.{layer}"

            # postfix dict
            postfix_mapping = eval(enc_dec + "_postfix_mapping")

            for baseline_postfix, my_postfix in postfix_mapping.items():
                # print(baseline_postfix, my_postfix)
                if type(my_postfix) == list:  # a baseline layer maps to multiple my layers
                    baseline_layer_name = ".".join([baseline_prefix, baseline_postfix])
                    my_layer_names = [".".join([my_prefix, p]) for p in my_postfix]

                    baseline_layer = baseline_sd.pop(baseline_layer_name)
                    my_layers = [my_sd.pop(my_layer_name) for my_layer_name in my_layer_names]

                    assert baseline_layer.shape == torch.cat(my_layers, dim=0).shape
                else:
                    baseline_layer_name = ".".join([baseline_prefix, baseline_postfix])
                    my_layer_name = ".".join([my_prefix, my_postfix])

                    baseline_layer = baseline_sd.pop(baseline_layer_name)
                    my_layer = my_sd.pop(my_layer_name)

                    assert baseline_layer.shape == my_layer.shape

        print(f"{enc_dec} transformer layers test pass")

        # check layernorm at the last
        for weight_bias in ["weight", "bias"]:
            baseline_layer_name = f"transformer.{enc_dec}.norm.{weight_bias}"
            my_layer_name = f"{enc_dec}.norm.{weight_bias}"

            baseline_layer = baseline_sd.pop(baseline_layer_name)
            my_layer = my_sd.pop(my_layer_name)

            assert baseline_layer.shape == my_layer.shape

    print("encoder/decoder test pass")

    # check for classifier
    for baseline_layer_name, my_layer_name in classifier_mapping.items():
        baseline_layer = baseline_sd.pop(baseline_layer_name)
        my_layer = my_sd.pop(my_layer_name)
        assert baseline_layer.shape == my_layer.shape
    print("classifier test pass")

    # check if both state_dicts are empty
    assert len(baseline_sd) == len(my_sd) == 0


def copy_state_dict(src, tgt):
    # src = baseline
    # tgt = mine

    copy_count = 0

    # copy embedding, classifier
    mappings = [embedding_mapping, classifier_mapping]
    for mapping in mappings:
        for src_layer_name, tgt_layer_name in mapping.items():
            if same_size(src[src_layer_name], tgt[tgt_layer_name]):
                tgt[tgt_layer_name] = src[src_layer_name]
                copy_count += 1
            else:
                assert False, "%d - %d tensor shape not matched" % (src_layer_name, tgt_layer_name)
    print("embedding and classifier copied")

    # check for transformer blocks
    for enc_dec in ["encoder", "decoder"]:
        # for every layer
        for layer in range(cfg.n_layers):
            # prefix
            src_prefix = f"transformer.{enc_dec}.layers.{layer}"
            tgt_prefix = f"{enc_dec}.layers.{layer}"

            # postfix dict
            postfix_mapping = eval(enc_dec + "_postfix_mapping")

            for src_postfix, tgt_postfix in postfix_mapping.items():
                if type(tgt_postfix) == list:  # a baseline layer maps to multiple my layers
                    src_layer_name = ".".join([src_prefix, src_postfix])
                    tgt_layer_names = [".".join([tgt_prefix, p]) for p in tgt_postfix]

                    src_concat_weight = src[src_layer_name]  # shape: (d_model * 3, d_model) for weights
                    src_weights = torch.split(src_concat_weight, src_concat_weight.size(0) // 3, dim=0)

                    for src_weight, tgt_layer_name in zip(src_weights, tgt_layer_names):
                        if same_size(tgt[tgt_layer_name], src_weight):
                            tgt[tgt_layer_name] = src_weight
                            copy_count += 1
                        else:
                            assert False, "%d - %d tensor shape not matched" % (src_layer_name, tgt_layer_name)
                else:
                    src_layer_name = ".".join([src_prefix, src_postfix])
                    tgt_layer_name = ".".join([tgt_prefix, tgt_postfix])

                    if same_size(src[src_layer_name], tgt[tgt_layer_name]):
                        tgt[tgt_layer_name] = src[src_layer_name]
                        copy_count += 1
                    else:
                        assert False, "%d - %d tensor shape not matched" % (src_layer_name, tgt_layer_name)

        print(f"{enc_dec} transformer layers copied")

        # check layernorm at the last
        for weight_bias in ["weight", "bias"]:
            src_layer_name = f"transformer.{enc_dec}.norm.{weight_bias}"
            tgt_layer_name = f"{enc_dec}.norm.{weight_bias}"

            if same_size(src[src_layer_name], tgt[tgt_layer_name]):
                tgt[tgt_layer_name] = src[src_layer_name]
                copy_count += 1
            else:
                assert False, "%d - %d tensor shape not matched" % (src_layer_name, tgt_layer_name)

    assert (copy_count == len(tgt)), "not all weights are copied"

    return tgt

    # check if both state_dicts are empty
    # assert len(baseline_sd) == len(my_sd) == 0


def test_transformer():
    print("device:", cfg.device)
    torch.manual_seed(0)
    # # fairseq model
    # # List available models
    # torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]
    #
    # # Load a transformer trained on WMT'16 En-De
    # # Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
    # baseline_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt16.en-de',
    #                        tokenizer='moses', bpe='subword_nmt')
    # baseline_model.to(cfg.device)

    # torch tutorial model
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

    # # 1. check # of params
    # print("1. check the number of params")
    # # assert count_params(baseline_model) == count_params(my_model), \
    # #     "param number mismatch-> baseline: %d, mine: %d" % (count_params(baseline_model), count_params(my_model))
    # print("total # of params of baseline model:", count_params(baseline_model))
    # print("total # of params of my model:", count_params(my_model))

    # # 2. check output shape
    # print("\n2. check model output shape")
    # baseline_out = baseline_model(enc_input, dec_input)
    # my_out = my_model(enc_input, dec_input)
    # # assert baseline_output.shape == my_out.shape, \
    # #     "output shape mismatch-> baseline: %d, mine: %d" % (baseline_output.shape, my_out.shape)
    # print("baseline model output shape:", baseline_out.shape)
    # print("my model output shape:", my_out.shape)

    # # 3. compare state_dict
    # print("\n3. check model parameters shape")
    # baseline_sd = copy.deepcopy(baseline_model.state_dict())
    # my_sd = copy.deepcopy(my_model.state_dict())
    # check_state_dict(baseline_sd, my_sd)

    # print state dict shape
    # for k, v in baseline_model.state_dict().items():
    #     print(k, "shape:", v.shape)
    # print()
    # for k, v in my_model.state_dict().items():
    #     print(k, "shape:", v.shape)

    # 4. test masking
    # tests/test_mask_baseline.py

    # 5. copy state dict
    print("\n4. copy model parameters baseline -> mine")
    baseline_sd = copy.deepcopy(baseline_model.state_dict())
    my_sd = copy.deepcopy(my_model.state_dict())
    copied_my_sd = copy_state_dict(src=baseline_sd,
                                   tgt=my_sd)

    # define new copied model
    my_model_copy = type(my_model)(n_layers=cfg.n_layers,
                                   d_model=cfg.d_model,
                                   h=cfg.h,
                                   ffn_hidden=cfg.ffn_hidden,
                                   p_drop=cfg.p_drop,
                                   src_vocab_size=cfg.src_vocab_size,
                                   tgt_vocab_size=cfg.tgt_vocab_size,
                                   pad_idx=cfg.pad_idx,
                                   device=cfg.device)  # get a new instance
    my_model_copy.load_state_dict(copied_my_sd)
    my_model_copy.to(cfg.device)
    my_model_copy.eval()

    # test if baseline and my_copied weights are the same
    assert torch.isclose(my_model_copy.state_dict()['src_tok_emb.embedding.weight'], \
                         baseline_model.state_dict()['src_tok_emb.embedding.weight']).all()

    print("\n4-1. feed the input without padding")
    # compare the output
    # input
    enc_input = torch.randint(cfg.pad_idx + 1, cfg.src_vocab_size, (cfg.batch_size, cfg.src_seq_len)).to(cfg.device)
    dec_input = torch.randint(cfg.pad_idx + 1, cfg.tgt_vocab_size, (cfg.batch_size, cfg.tgt_seq_len)).to(cfg.device)

    # feed forward
    print("=======baseline feed forward ========")
    baseline_src_emb, baseline_tgt_emb, baseline_enc_out, baseline_dec_out, baseline_out = baseline_model(enc_input,
                                                                                                          dec_input)

    print("=======my feed forward ========")
    my_copy_src_emb, my_copy_enc_out, my_copy_tgt_emb, my_copy_dec_out, my_copy_out = my_model_copy(enc_input,
                                                                                                    dec_input)
    assert torch.isclose(my_copy_out, baseline_out,
                         atol=1e-6).all(), "(without padding) output of baseline and my model are different"
    print("(without padding) baseline and mine outputs are the same")

    print("\n4-2. feed the input with padding")
    # new input with padding
    src_padding_length = 5
    tgt_padding_length = 7
    src_padding = torch.Tensor([cfg.pad_idx] * src_padding_length).long()
    tgt_padding = torch.Tensor([cfg.pad_idx] * tgt_padding_length).long()

    src_input_nopad = torch.randint(cfg.pad_idx + 1, cfg.src_vocab_size,
                                    (cfg.batch_size, cfg.src_seq_len - src_padding_length))
    tgt_input_nopad = torch.randint(cfg.pad_idx + 1, cfg.tgt_vocab_size,
                                    (cfg.batch_size, cfg.tgt_seq_len - tgt_padding_length))

    src_input_pad = torch.cat((src_input_nopad, src_padding.expand(cfg.batch_size, -1)), dim=1).to(cfg.device)
    tgt_input_pad = torch.cat((tgt_input_nopad, tgt_padding.expand(cfg.batch_size, -1)), dim=1).to(cfg.device)

    # fix input
    src_input_pad = torch.LongTensor([[6, 3, 5, 4, 1, 1, 1, 1, 1, 1],
                                      [4, 9, 4, 6, 7, 4, 8, 1, 1, 1]]).to(cfg.device)
    tgt_input_pad = torch.LongTensor([[3, 7, 14, 5, 7, 0, 4, 3, 1, 1, 1, 1],
                                      [13, 2, 8, 9, 7, 1, 1, 1, 1, 1, 1, 1]]).to(cfg.device)
    # compare the output
    baseline_src_emb, baseline_tgt_emb, baseline_enc_out, baseline_dec_out, baseline_out = baseline_model(src_input_pad,
                                                                                                          tgt_input_pad)
    my_copy_src_emb, my_copy_enc_out, my_copy_tgt_emb, my_copy_dec_out, my_copy_out = my_model_copy(src_input_pad,
                                                                                                    tgt_input_pad)

    # src embedding, tgt embedding are the same
    # outputs are nan - check
    breakpoint()
    assert torch.isclose(baseline_out,
                         my_copy_out).all(), "(with padding) output of baseline and my model are different"
    print("(with padding) baseline and my outputs are the same")


if __name__ == "__main__":
    test_transformer()
