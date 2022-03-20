"""
Case 1. Test an encoder layer for unpadded input

baseline: torch.nn.TransformerEncoderLayer
my: model.layer.EncoderLayer

a) float32
embedding output - always the same
encoder output - differs sometimes (for small numbers?), atol = 1e-6 or 1e-7

b) float64 = double
embedding output, encoder_output - always the same
"""

import copy

from model.sublayer.positional_encoding import PositionalEncoding as my_PE
from model.sublayer.token_embedding import TokenEmbedding as my_TE
from baseline_model.baseline_transformer import PositionalEncoding as baseline_PE
from baseline_model.baseline_transformer import TokenEmbedding as baseline_TE

from torch.nn import TransformerEncoderLayer as BaselineEncLayer
from model.layer.encoder_layer import EncoderLayer as MyEncLayer

from test_utils import *

# global variables
load_input_flag = False  # True if to load stored input tensor
save_input_flag = False  # True if to save generated random input tensor
input_file_path = 'test_input/input_nopad_2by10.pt'  # file path to the stored input tensor
seed = 3
double_flag = False  # True if to test with double dtype


def test_encoder_nopad():
    # manual seed
    torch.manual_seed(seed)

    module_list = []

    # baseline model
    # 1. embedding
    baseline_src_te = baseline_TE(vocab_size=cfg.src_vocab_size, emb_size=cfg.d_model)
    baseline_pe = baseline_PE(emb_size=cfg.emb_size, dropout=cfg.p_drop)

    module_list.append(baseline_src_te)
    module_list.append(baseline_pe)

    # 2. encoder layer
    baseline_enc_layer = BaselineEncLayer(d_model=cfg.d_model,
                                          nhead=cfg.h,
                                          dim_feedforward=cfg.ffn_hidden,
                                          dropout=cfg.p_drop,
                                          batch_first=True,
                                          device=cfg.device)
    module_list.append(baseline_enc_layer)

    # my model
    # 1. embedding
    my_src_te = my_TE(emb_size=cfg.d_model, vocab_size=cfg.src_vocab_size)
    my_pe = my_PE(emb_size=cfg.d_model, p_drop=cfg.p_drop)

    module_list.append(my_src_te)
    module_list.append(my_pe)

    # 2. encoder layer
    my_enc_layer = MyEncLayer(d_model=cfg.d_model,
                              h=cfg.h,
                              ffn_hidden=cfg.ffn_hidden,
                              p_drop=cfg.p_drop)
    module_list.append(my_enc_layer)

    # load module to device, make them eval
    # if needed, convert models to double
    for module in module_list:
        module.to(cfg.device)
        module.eval()
        if double_flag: module.double()

    # copy state_dict
    # 1-1. token embedding
    copied_my_src_te_sd = copy_te_dict(src=baseline_src_te.state_dict(),
                                       tgt=copy.deepcopy(my_src_te.state_dict()))
    my_src_te.load_state_dict(copied_my_src_te_sd)
    # assert torch.isclose(baseline_src_te.state_dict()["embedding.weight"], my_src_te.state_dict()["embedding.weight"]).all().item()

    # 1-2. positional encoding
    copied_my_pe_sd = copy_pe_dict(src=baseline_pe.state_dict(),
                                   tgt=copy.deepcopy(my_pe.state_dict()))
    my_pe.load_state_dict(copied_my_pe_sd)
    # assert torch.isclose(baseline_pe.state_dict()["pos_embedding"], my_pe.state_dict()["encoding"]).all().item()

    # 2. encoder layer
    copy_my_enc_sd = copy_transformer_layer_dict(src=baseline_enc_layer.state_dict(),
                                                 tgt=copy.deepcopy(my_enc_layer.state_dict()), 
                                                 model_type="encoder")
    my_enc_layer.load_state_dict(copy_my_enc_sd)

    # feed an input
    print("=" * 30, "feed an input without padding", "=" * 30)
    if load_input_flag:  # load saved input
        input_nopad = load_input_nopad(input_file_path).to(cfg.device)
    else:
        input_nopad = generate_input_nopad(seed=seed,
                                           pad_idx=cfg.pad_idx,
                                           vocab_size=cfg.src_vocab_size,
                                           seq_len=cfg.src_seq_len,
                                           batch_size=cfg.batch_size,
                                           save_flag=save_input_flag,
                                           file_path=input_file_path).to(cfg.device)  # generate rand input

    # embedding output, shape: (batch_size,src_seq_len, d_model)
    baseline_src_emb = baseline_pe(baseline_src_te(input_nopad))
    my_src_emb = my_pe(my_src_te(input_nopad))
    emb_flag = torch.isclose(baseline_src_emb, my_src_emb).all().item()
    print("Embedding outputs are the same?:", emb_flag)

    # encoder layer output, shape: (batch_size, src_seq_len, d_model)
    baseline_enc_out = baseline_enc_layer(baseline_src_emb)
    my_enc_out = my_enc_layer(my_src_emb)

    # compare attn output
    attn_out_flag = torch.isclose(baseline_enc_layer.attn_out, my_enc_layer.attn_out).all().item()
    print("Attention outputs are the same?:", attn_out_flag)

    # compare attn score
    attn_score_flag = torch.isclose(baseline_enc_layer.attn_score,
                                    torch.mean(my_enc_layer.attn_score, dim=1)).all().item()
    print("Attention scores are the same?:", attn_score_flag)

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
        exit


if __name__ == "__main__":
    # input_nopad = generate_input_nopad(seed=seed,
    #                                    pad_idx=cfg.pad_idx,
    #                                    vocab_size=cfg.src_vocab_size,
    #                                    seq_len=cfg.src_seq_len,
    #                                    batch_size=cfg.batch_size,
    #                                    save_flag=save_input_flag,
    #                                    file_path=input_file_path)
    test_encoder_nopad()
