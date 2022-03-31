"""
Test an decoder layer for padded input

baseline: torch.nn.TransformerDecoderLayer
my: model.layer.DecoderLayer

"""

import copy

from model.sublayer.positional_encoding import PositionalEncoding as my_PE
from model.sublayer.token_embedding import TokenEmbedding as my_TE
from baseline_model.baseline_transformer import PositionalEncoding as baseline_PE
from baseline_model.baseline_transformer import TokenEmbedding as baseline_TE

from torch.nn import TransformerDecoderLayer as BaselineDecLayer
from model.layer.decoder_layer import DecoderLayer as MyDecLayer

from test_utils import *
import config as cfg

# global variables
print_input_flag = True
load_input_flag = False  # True if to load stored input tensor
save_input_flag = False  # True if to save generated random input tensor
input_file_path = 'test_input/tgt_input_pad_2by12.pt'  # file path to the stored input tensor
seed = 0
double_flag = False  # True if to test with double dtype


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_baseline_tgt_mask(tgt, device, pad_idx):  # changed to batch_first
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    tgt_padding_mask = (tgt == pad_idx)
    return tgt_mask, tgt_padding_mask


def create_my_tgt_pad_masks(x, pad_idx):  # creates padding mask
    """
    Create a mask to hide padding

    :param x: torch.Tensor, shape: (batch_size, seq_len)
    :param pad_idx: int, pad token index

    :return: torch.Tensor, shape: (batch_size, seq_len)
    """
    return x != pad_idx


def create_ag_mask(x: Tensor) -> Tensor:
    """
    :param x: torch.Tensor, shape: (batch_size, seq_len)
    :return: torch.Tensor, shape: (batch_size, seq_len, seq_len)
    """
    seq_len = x.size(1)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=cfg.device).tril(diagonal=0)
    return mask


def test_decoder_pad():
    # manual seed
    torch.manual_seed(seed)

    module_list = []

    # baseline model
    # 1. embedding
    baseline_tgt_te = baseline_TE(vocab_size=cfg.tgt_vocab_size, emb_size=cfg.d_model)
    baseline_pe = baseline_PE(emb_size=cfg.emb_size, dropout=cfg.p_drop)

    module_list.append(baseline_tgt_te)
    module_list.append(baseline_pe)

    # 2. decoder layer
    baseline_dec_layer = BaselineDecLayer(d_model=cfg.d_model,
                                          nhead=cfg.h,
                                          dim_feedforward=cfg.ffn_hidden,
                                          dropout=cfg.p_drop,
                                          batch_first=True,
                                          device=cfg.device)
    module_list.append(baseline_dec_layer)

    # my model
    # 1. embedding
    my_tgt_te = my_TE(emb_size=cfg.d_model, vocab_size=cfg.tgt_vocab_size)
    my_pe = my_PE(emb_size=cfg.d_model, p_drop=cfg.p_drop)

    module_list.append(my_tgt_te)
    module_list.append(my_pe)

    # 2. decoder layer
    my_dec_layer = MyDecLayer(d_model=cfg.d_model,
                              h=cfg.h,
                              ffn_hidden=cfg.ffn_hidden,
                              p_drop=cfg.p_drop)
    module_list.append(my_dec_layer)

    # load module to device, make them eval
    # if needed, convert models to double
    for module in module_list:
        module.to(cfg.device)
        module.eval()
        if double_flag: module.double()

    # copy state_dict
    # 1-1. token embedding
    copied_my_tgt_te_sd = copy_te_dict(src=baseline_tgt_te.state_dict(),
                                       tgt=copy.deepcopy(my_tgt_te.state_dict()))
    my_tgt_te.load_state_dict(copied_my_tgt_te_sd)
    assert torch.isclose(baseline_tgt_te.state_dict()["embedding.weight"],
                         my_tgt_te.state_dict()["embedding.weight"]).all().item()

    # 1-2. positional encoding
    copied_my_pe_sd = copy_pe_dict(src=baseline_pe.state_dict(),
                                   tgt=copy.deepcopy(my_pe.state_dict()))
    my_pe.load_state_dict(copied_my_pe_sd)
    assert torch.isclose(baseline_pe.state_dict()["pos_embedding"], my_pe.state_dict()["encoding"]).all().item()

    # 2. decoder layer
    copy_my_dec_sd = copy_transformer_layer_dict(src=baseline_dec_layer.state_dict(),
                                                 tgt=copy.deepcopy(my_dec_layer.state_dict()),
                                                 model_type="decoder")
    my_dec_layer.load_state_dict(copy_my_dec_sd)

    # feed an input
    print("=" * 30, "feed an input with padding", "=" * 30)
    if load_input_flag:  # load saved input
        input_pad = load_input_pad(input_file_path).to(cfg.device)
    else:
        input_pad = generate_input_pad(seed=seed,
                                       pad_idx=cfg.pad_idx,
                                       vocab_size=cfg.tgt_vocab_size,
                                       seq_len=cfg.tgt_seq_len,
                                       batch_size=cfg.batch_size,
                                       print_flag=print_input_flag,
                                       save_flag=save_input_flag,
                                       file_path=input_file_path).to(cfg.device)  # generate rand input

    # create masks
    baseline_tgt_mask, baseline_tgt_pad_mask = create_baseline_tgt_mask(tgt=input_pad,
                                                                        device=cfg.device,
                                                                        pad_idx=cfg.pad_idx)
    baseline_memory_mask = None
    src_valid_token_count = cfg.src_seq_len // 2
    src_pad_token_count = cfg.src_seq_len - src_valid_token_count
    baseline_memory_key_padding_mask = torch.BoolTensor([[False] * src_valid_token_count + [True] * src_pad_token_count] * cfg.batch_size).to(cfg.device)

    my_tgt_ag_mask = create_ag_mask(input_pad)
    my_tgt_pad_mask = create_my_tgt_pad_masks(x=input_pad, pad_idx=cfg.pad_idx)
    my_memory_pad_mask = ~baseline_memory_key_padding_mask

    # check: baseline_tgt_mask is float, not bool
    assert torch.isclose(baseline_tgt_pad_mask, ~my_tgt_pad_mask).all()
    assert torch.isclose(baseline_tgt_mask.bool(), ~my_tgt_ag_mask).all()

    # embedding output, shape: (batch_size, tgt_seq_len, d_model)
    baseline_tgt_emb = baseline_pe(baseline_tgt_te(input_pad))
    my_tgt_emb = my_pe(my_tgt_te(input_pad))
    emb_flag = torch.isclose(baseline_tgt_emb, my_tgt_emb).all().item()
    print("Embedding outputs are the same?:", emb_flag)

    # decoder layer output, shape: (batch_size, tgt_seq_len, d_model)
    enc_output = torch.rand((cfg.batch_size, cfg.src_seq_len, cfg.d_model)).to(cfg.device)
    baseline_dec_out = baseline_dec_layer(tgt=baseline_tgt_emb,
                                          memory=enc_output,
                                          tgt_mask=baseline_tgt_mask,
                                          tgt_key_padding_mask=baseline_tgt_pad_mask,
                                          memory_mask=baseline_memory_mask,
                                          memory_key_padding_mask=baseline_memory_key_padding_mask)

    my_dec_out = my_dec_layer(tgt_emb=my_tgt_emb,
                              enc_output=enc_output,
                              tgt_pad_mask=my_tgt_pad_mask,
                              tgt_autoregressive_mask=my_tgt_ag_mask,
                              memory_pad_mask=my_memory_pad_mask,)

    # compare self attn output
    self_attn_out_flag = torch.isclose(baseline_dec_layer.self_attn_out, my_dec_layer.self_attn_out).all().item()
    print("Self attention outputs are the same?:", self_attn_out_flag)

    # compare self attn score
    cross_attn_score_flag = torch.isclose(baseline_dec_layer.self_attn_score, my_dec_layer.self_attn_score).all().item()
    print("Self attention scores are the same?:", cross_attn_score_flag)

    # compare cross attn output
    cross_attn_out_flag = torch.isclose(baseline_dec_layer.cross_attn_out, my_dec_layer.cross_attn_out).all().item()
    print("cross attention outputs are the same?:", cross_attn_out_flag)

    # compare cross attn score
    cross_attn_score_flag = torch.isclose(baseline_dec_layer.cross_attn_score, my_dec_layer.cross_attn_score).all().item()
    print("cross attention scores are the same?:", cross_attn_score_flag)

    # compare decoder output
    decoder_flag = torch.isclose(baseline_dec_out, my_dec_out).all().item()
    print("decoder outputs are the same?:", decoder_flag)

    if not decoder_flag:
        unmatch_indices = (torch.isclose(baseline_dec_out, my_dec_out) == False).nonzero(as_tuple=True)
        baseline_unmatch = baseline_dec_out[unmatch_indices]
        my_unmatch = my_dec_out[unmatch_indices]

        atol = 1e-6  # or 1e-7
        print("unmatched count:", len(baseline_unmatch))
        print(f"compare with low precision ({atol}):", torch.isclose(baseline_unmatch, my_unmatch, atol=atol))

        breakpoint()
    else:
        print("Test Successful!")


if __name__ == "__main__":
    # generate_input_pad(seed=seed,
    #                    pad_idx=cfg.pad_idx,
    #                    vocab_size=cfg.tgt_vocab_size,
    #                    seq_len=cfg.tgt_seq_len,
    #                    batch_size=cfg.batch_size,
    #                    save_flag=save_input_flag,
    #                    file_path=input_file_path)
    test_decoder_pad()
