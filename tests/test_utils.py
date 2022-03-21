import torch
import os
import numpy as np
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

def count_params(model):
    """
    count model parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in model_parameters])
    return param_count


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


def copy_encoder_decoder_dict(src, tgt, n_layers, model_type):
    # src = baseline
    # tgt = mine
    # type - "encoder" or "decoder"

    copy_count = 0

    for layer in range(n_layers):
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


def copy_transformer_dict(src, tgt, n_layers):
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
                assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
    print("embedding and classifier copied")

    # check for transformer blocks
    for enc_dec in ["encoder", "decoder"]:
        # for every layer
        for layer in range(n_layers):
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
                            assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"
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
                assert False, f"{src_layer_name} - {tgt_layer_name} doesn't match"

    assert len(tgt) == copy_count, "Transformer is not copied correctly"
    print("Transformer copied: baseline -> my")
    return tgt


def check_transformer_grads(src, tgt, atol=1e-7):
    # src = baseline
    # tgt = mine

    # make target parameter dict
    tgt_grads_dict = {}
    for tgt_layer_name, tgt_param in tgt.named_parameters():
        tgt_grads_dict[tgt_layer_name] = tgt_param.grad

    for src_layer_name, src_param in src.named_parameters():
        # gradient tensor
        src_grad = src_param.grad
        src_name_split = src_layer_name.split(".")

        # check layer type
        is_transformer = src_name_split[0] == "transformer"
        is_generator = src_name_split[0] == "generator"

        if is_transformer:  # transformer layer
            enc_dec = src_name_split[1]
            is_layer = src_name_split[2] == "layers"
            is_norm = src_name_split[2] == "norm"

            postfix_mapping = eval(enc_dec + "_postfix_mapping")

            if is_layer:
                layer_num = src_name_split[3]
                src_postfix = ".".join(src_name_split[4:])
                tgt_prefix = f"{enc_dec}.layers.{layer_num}"
                tgt_postfix = postfix_mapping[src_postfix]

                if type(tgt_postfix) == list:  # a baseline layer maps to multiple my layers

                    tgt_layer_names = [".".join([tgt_prefix, p]) for p in tgt_postfix]

                    src_grad_list = torch.split(src_grad, src_grad.size(0) // 3, dim=0)

                    for src_grad_split, tgt_layer_name in zip(src_grad_list, tgt_layer_names):
                        tgt_grad = tgt_grads_dict.pop(tgt_layer_name)
                        if not torch.isclose(src_grad_split, tgt_grad).all():
                            print(f"{src_layer_name} - {tgt_layer_name} doesn't match")
                            unmatch_indices = (torch.isclose(src_grad_split, tgt_grad) == False).nonzero(as_tuple=True)
                            src_unmatch = src_grad_split[unmatch_indices]
                            tgt_unmatch = tgt_grad[unmatch_indices]

                            print("unmatched count:", len(src_unmatch))
                            print("unmatched values:", src_unmatch)
                            print(f"compare with low precision ({atol}):",
                                  torch.isclose(src_unmatch, tgt_unmatch, atol=atol))
                            print()
                        else:
                            # print(f"{src_layer_name} - {tgt_layer_name} match")
                            pass

                else:
                    tgt_layer_name = ".".join([tgt_prefix, tgt_postfix])
                    tgt_grad = tgt_grads_dict.pop(tgt_layer_name)

                    if not torch.isclose(src_grad, tgt_grad).all():
                        print(f"{src_layer_name} - {tgt_layer_name} doesn't match")
                        unmatch_indices = (torch.isclose(src_grad, tgt_grad) == False).nonzero(as_tuple=True)
                        src_unmatch = src_grad[unmatch_indices]
                        tgt_unmatch = tgt_grad[unmatch_indices]

                        print("unmatched count:", len(src_unmatch))
                        print("unmatched values:", src_unmatch)
                        print(f"compare with low precision ({atol}):",
                              torch.isclose(src_unmatch, tgt_unmatch, atol=atol))
                        print()
                    else:
                        # print(f"{src_layer_name} - {tgt_layer_name} match")
                        pass
                    # or just use
                    # assert torch.isclose(src_grad, tgt_grad).all(), f"{src_layer_name} - {tgt_layer_name} doesn't match"

            elif is_norm:  # layernorm after transformer layer
                postfix_mapping = layernorm_mapping
                src_postfix = ".".join(src_name_split[2:])
                tgt_prefix = f"{enc_dec}"
                tgt_postfix = postfix_mapping[src_postfix]

                tgt_layer_name = ".".join([tgt_prefix, tgt_postfix])
                tgt_grad = tgt_grads_dict.pop(tgt_layer_name)

                if not torch.isclose(src_grad, tgt_grad).all():
                    print(f"{src_layer_name} - {tgt_layer_name} doesn't match")
                    unmatch_indices = (torch.isclose(src_grad, tgt_grad) == False).nonzero(as_tuple=True)
                    src_unmatch = src_grad[unmatch_indices]
                    tgt_unmatch = tgt_grad[unmatch_indices]

                    print("unmatched count:", len(src_unmatch))
                    print("unmatched values:", src_unmatch)
                    print(f"compare with low precision ({atol}):",
                          torch.isclose(src_unmatch, tgt_unmatch, atol=atol))
                    print()
                else:
                    # print(f"{src_layer_name} - {tgt_layer_name} match")
                    pass
            else:
                assert False, f"invalid layer type - {src_layer_name}"

        elif is_generator:
            postfix_mapping = classifier_mapping
            tgt_layer_name = postfix_mapping[src_layer_name]

            tgt_grad = tgt_grads_dict.pop(tgt_layer_name)

            if not torch.isclose(src_grad, tgt_grad).all():
                print(f"{src_layer_name} - {tgt_layer_name} doesn't match")
                unmatch_indices = (torch.isclose(src_grad, tgt_grad) == False).nonzero(as_tuple=True)
                src_unmatch = src_grad[unmatch_indices]
                tgt_unmatch = tgt_grad[unmatch_indices]

                print("unmatched count:", len(src_unmatch))
                print("unmatched values:", src_unmatch)
                print(f"compare with low precision ({atol}):",
                      torch.isclose(src_unmatch, tgt_unmatch, atol=atol))
                print()
            else:
                # print(f"{src_layer_name} - {tgt_layer_name} match")
                pass

        else:  # embedding
            postfix_mapping = embedding_mapping
            tgt_layer_name = postfix_mapping[src_layer_name]

            tgt_grad = tgt_grads_dict.pop(tgt_layer_name)

            if not torch.isclose(src_grad, tgt_grad).all():
                print(f"{src_layer_name} - {tgt_layer_name} doesn't match")
                unmatch_indices = (torch.isclose(src_grad, tgt_grad) == False).nonzero(as_tuple=True)
                src_unmatch = src_grad[unmatch_indices]
                tgt_unmatch = tgt_grad[unmatch_indices]

                print("unmatched count:", len(src_unmatch))
                print("unmatched values:", src_unmatch)
                print(f"compare with low precision ({atol}):",
                      torch.isclose(src_unmatch, tgt_unmatch, atol=atol))
                print()
            else:
                # print(f"{src_layer_name} - {tgt_layer_name} match")
                pass
    assert len(tgt_grads_dict) == 0, "unchecked parameter left in target dictionary"
    print("Gradients checked!")
    return

