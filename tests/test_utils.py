import torch
import typing
import os
import config as cfg
from layer_mapping import *

from torch import Tensor, LongTensor


def generate_input_nopad(seed: int,
                         pad_idx: int,
                         vocab_size: int,
                         seq_len: int,
                         batch_size: int,
                         save_flag: bool = False,
                         file_path: str = None) -> LongTensor:
    """
    generate random input tensor with no padding. save it if save_flag=True
    :param seed: int, random seed
    :param save_flag: bool, whether to save the generated tensor
    :param file_path: str, path to save the generated tensor
    :return input_nopad: LongTensor, shape: (batch_size, src_seq_len)
    """

    torch.manual_seed(seed)
    input_nopad = torch.randint(pad_idx + 1, vocab_size, (batch_size, seq_len)).long()
    print("input_nopad shape:", input_nopad.shape)
    print("input_nopad:", input_nopad)

    if save_flag:
        current_path = os.path.dirname(os.path.realpath(__file__))
        torch.save(input_nopad, os.path.join(current_path, file_path))
        print("input tensor saved to", file_path)

    return input_nopad


def load_input_nopad(file_path: str):
    """
    load an input tensor

    :param file_path: path to the saved tensor
    :return input_nopad: LongTensor, shape: (batch_size, src_seq_len)
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    input_nopad = torch.load(os.path.join(current_path, file_path))
    print("input tensor loaded from", file_path)
    print("input_nopad shape:", input_nopad.shape)
    print("input_nopad:", input_nopad)
    return input_nopad


def generate_input_pad(seed: int,
                       pad_idx: int,
                       vocab_size: int,
                       seq_len: int,
                       batch_size: int,
                       save_flag: bool = False,
                       file_path: str = None) -> LongTensor:
    """
    generate random input tensor with padding. save it if save_flag=True
    :param seed: int, random seed
    :param save_flag: bool, whether to save the generated tensor
    :param file_path: str, path to save the generated tensor
    :return input_pad: LongTensor, shape: (batch_size, src_seq_len)
    """
    torch.manual_seed(seed)
    valid_token_counts = torch.randint(1, seq_len,(batch_size,))  # number of valid tokens (not <pad>) in each input

    input_pad = torch.empty(size=(batch_size, seq_len), dtype=torch.long)
    for i in range(batch_size):
        valid_length = valid_token_counts[i]
        pad_count = seq_len - valid_length

        valid = torch.randint(pad_idx + 1, vocab_size, (valid_length,))  # make 1d tensor (one input)
        padding = torch.LongTensor([pad_idx] * pad_count)
        sample = torch.cat((valid, padding), dim=0)
        input_pad[i] = sample

    print("input_pad shape:", input_pad.shape)
    print("input_pad:", input_pad)

    if save_flag:
        current_path = os.path.dirname(os.path.realpath(__file__))
        torch.save(input_pad, os.path.join(current_path, file_path))
        print("input tensor saved to", file_path)

    return input_pad


def load_input_pad(file_path):
    current_path = os.path.dirname(os.path.realpath(__file__))
    input_pad = torch.load(os.path.join(current_path, file_path))
    print("input tensor loaded from", file_path)
    print("input_pad shape:", input_pad.shape)
    print("input_pad:", input_pad)
    return input_pad


def same_size(x: Tensor, y: Tensor) -> bool:
    """
    check if x and y has the same size
    """
    return x.shape == y.shape


# model weights copy methods
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


def copy_transformer_layer_dict(src, tgt, model_type):
    copy_count = 0
    # postfix dict
    postfix_mapping = eval(model_type + "_postfix_mapping")

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
    assert len(tgt) == copy_count, f"{model_type} layer is not copied correctly"
    print(f"{model_type} layer copied: baseline -> my")
    return tgt


def copy_encoder_decoder_dict(src, tgt, model_type):
    # src = baseline
    # tgt = mine
    # type - "encoder" or "decoder"

    copy_count = 0

    for layer in range(cfg.n_layers):
        # prefix
        src_prefix = f"layers.{layer}"
        tgt_prefix = f"layers.{layer}"

        # postfix dict
        postfix_mapping = eval(model_type + "_postfix_mapping")

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
                        assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
            else:
                src_layer_name = ".".join([src_prefix, src_postfix])
                tgt_layer_name = ".".join([tgt_prefix, tgt_postfix])

                if same_size(src[src_layer_name], tgt[tgt_layer_name]):
                    tgt[tgt_layer_name] = src[src_layer_name]
                    copy_count += 1
                else:
                    assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"

    print(f"{model_type} transformer layers copied")

    # check layernorm at the last
    for weight_bias in ["weight", "bias"]:
        src_layer_name = f"norm.{weight_bias}"
        tgt_layer_name = f"norm.{weight_bias}"

        if same_size(src[src_layer_name], tgt[tgt_layer_name]):
            tgt[tgt_layer_name] = src[src_layer_name]
            copy_count += 1
        else:
            assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"

    assert len(tgt) == copy_count, f"{model_type} is not copied correctly"
    print(f"{model_type} copied: baseline -> my")
    return tgt