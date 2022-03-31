"""
Test an encoder for padded input

baseline: torch.nn.TransformerEncoderLayer
my: model.layer.EncoderLayer

"""

import copy

from model.sublayer.positional_encoding import PositionalEncoding as my_PE
from model.sublayer.token_embedding import TokenEmbedding as my_TE
from baseline_model.baseline_transformer import PositionalEncoding as baseline_PE
from baseline_model.baseline_transformer import TokenEmbedding as baseline_TE

from torch.nn import LayerNorm
from torch.nn import TransformerEncoderLayer as BaselineEncLayer
from torch.nn import TransformerEncoder as BaselineEncoder
from model.encoder import Encoder as MyEncoder

from test_utils import *
import config as cfg

# global variables
print_input_flag = True
load_input_flag = False  # True if to load stored input tensor
save_input_flag = False  # True if to save generated random input tensor
input_file_path = 'test_input/input_pad_2by10.pt'  # file path to the stored input tensor
seed = 0
double_flag = False  # True if to test with double dtype


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

    :return: torch.Tensor, shape: (batch_size, seq_len)
    """
    return x != pad_idx


def test_encoder_pad():
    # manual seed
    torch.manual_seed(seed)

    module_list = []

    # baseline model
    # 1. embedding
    baseline_src_te = baseline_TE(vocab_size=cfg.src_vocab_size, emb_size=cfg.d_model)
    baseline_pe = baseline_PE(emb_size=cfg.emb_size, dropout=cfg.p_drop)

    module_list.append(baseline_src_te)
    module_list.append(baseline_pe)

    # 2. encoder
    baseline_enc_layer = BaselineEncLayer(d_model=cfg.d_model,
                                          nhead=cfg.h,
                                          dim_feedforward=cfg.ffn_hidden,
                                          dropout=cfg.p_drop,
                                          batch_first=True,
                                          device=cfg.device)
    baseline_norm = LayerNorm(cfg.d_model)
    baseline_encoder = BaselineEncoder(encoder_layer=baseline_enc_layer,
                                       num_layers=cfg.n_layers,
                                       norm=baseline_norm)
    module_list.append(baseline_encoder)

    # my model
    # 1. embedding
    my_src_te = my_TE(emb_size=cfg.d_model, vocab_size=cfg.src_vocab_size)
    my_pe = my_PE(emb_size=cfg.d_model, p_drop=cfg.p_drop)

    module_list.append(my_src_te)
    module_list.append(my_pe)

    # 2. encoder layer
    my_encoder = MyEncoder(n_layers=cfg.n_layers,
                           d_model=cfg.d_model,
                           h=cfg.h,
                           ffn_hidden=cfg.ffn_hidden,
                           p_drop=cfg.p_drop)
    module_list.append(my_encoder)

    # load module to device, make them eval
    # if needed, convert models to double
    for module in module_list:
        module.to(cfg.device)
        module.eval()
        if double_flag: module.double()

    # check if models are correctly built
    # breakpoint()

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

    # 2. encoder
    copy_my_enc_sd = copy_encoder_decoder_dict(src=baseline_encoder.state_dict(),
                                               tgt=copy.deepcopy(my_encoder.state_dict()),
                                               n_layers=cfg.n_layers,
                                               model_type="encoder")
    my_encoder.load_state_dict(copy_my_enc_sd)

    # feed an input
    print("=" * 30, "feed an input with padding", "=" * 30)
    if load_input_flag:  # load saved input
        input_pad = load_input_pad(input_file_path).to(cfg.device)
    else:
        input_pad = generate_input_pad(seed=seed,
                                       pad_idx=cfg.pad_idx,
                                       vocab_size=cfg.src_vocab_size,
                                       seq_len=cfg.src_seq_len,
                                       batch_size=cfg.batch_size,
                                       print_flag=print_input_flag,
                                       save_flag=save_input_flag,
                                       file_path=input_file_path).to(cfg.device)  # generate rand input

    # create masks
    # baseline_src_mask shape: (src_seq_len, src_seq_len)
    # baseline_src_padding_mask shape: (batch_size, src_seq_len)
    # my_src_padding_mask shape: (batch_size, 1, src_seq_len)
    baseline_src_mask, baseline_src_padding_mask = create_baseline_src_masks(src=input_pad, device=cfg.device,
                                                                             pad_idx=cfg.pad_idx)
    my_src_padding_mask = create_my_src_masks(x=input_pad, pad_idx=cfg.pad_idx)

    # embedding output, shape: (batch_size,src_seq_len, d_model)
    baseline_src_emb = baseline_pe(baseline_src_te(input_pad))
    my_src_emb = my_pe(my_src_te(input_pad))
    emb_flag = torch.isclose(baseline_src_emb, my_src_emb).all().item()
    print("Embedding outputs are the same?:", emb_flag)

    # encoder layer output, shape: (batch_size, src_seq_len, d_model)
    my_enc_out = my_encoder(src_emb=my_src_emb,
                            src_pad_mask=my_src_padding_mask)
    baseline_enc_out = baseline_encoder(src=baseline_src_emb,
                                        mask=baseline_src_mask,
                                        src_key_padding_mask=baseline_src_padding_mask)

    # compare encoder output
    encoder_flag = torch.isclose(baseline_enc_out, my_enc_out).all().item()
    print("Encoder outputs are the same?:", encoder_flag)

    if not encoder_flag:
        unmatch_indices = (torch.isclose(baseline_enc_out, my_enc_out) == False).nonzero(as_tuple=True)
        baseline_unmatch = baseline_enc_out[unmatch_indices]
        my_unmatch = my_enc_out[unmatch_indices]

        atol = 1e-6  # or 1e-7
        print("unmatched count:", len(baseline_unmatch))
        print(f"compare with low precision ({atol}):", torch.isclose(baseline_unmatch, my_unmatch, atol=atol))

        breakpoint()
    else:
        print("Test Successful!")


if __name__ == "__main__":
    # generate_input_pad(seed, save_flag=save_input_flag, file_path=input_file_path)
    test_encoder_pad()
