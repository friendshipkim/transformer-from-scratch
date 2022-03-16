import config
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.sublayer.token_embedding import TokenEmbedding
from model.sublayer.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
            self,
            n_layers: int,
            d_model: int,
            h: int,
            ffn_hidden: int,
            p_drop: float,
            src_vocab_size: int,
            tgt_vocab_size: int,
            pad_idx: int,
            device: str
    ):
        super(Transformer, self).__init__()

        self.pad_idx = pad_idx
        self.device = device
        # self.h = h  # TODO: remove not to use broadcasting

        # encoder
        self.src_tok_emb = TokenEmbedding(emb_size=d_model, vocab_size=src_vocab_size)
        self.positional_encoding = PositionalEncoding(emb_size=d_model, p_drop=p_drop)  # TODO: dropout necessary?
        self.encoder = Encoder(
            n_layers=n_layers,
            d_model=d_model,
            h=h,
            ffn_hidden=ffn_hidden,
            p_drop=p_drop,
        )

        # decoder
        self.tgt_tok_emb = TokenEmbedding(emb_size=d_model, vocab_size=tgt_vocab_size)
        # how to make shared embedding
        # 1. assign same weight when initialization
        # 2. self.src_tok_emb = self.tgt_tok_emb
        # what if source/target vocab size is different?
        self.decoder = Decoder(
            n_layers=n_layers,
            d_model=d_model,
            h=h,
            ffn_hidden=ffn_hidden,
            p_drop=p_drop,
        )

        # final classifier
        self.classifier = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        :param src: torch.Tensor, shape: (batch_size, src_seq_len)
        :param tgt: torch.Tensor, shape: (batch_size, tgt_seq_len)

        :return: torch.Tensor, shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # mask
        src_mask, cross_mask, _, _, tgt_mask = self.create_mask(src, tgt)

        # encoder
        enc_embedding = self.positional_encoding(self.src_tok_emb(src))
        enc_output = self.encoder(x=enc_embedding, enc_mask=src_mask)  # memory

        # decoder
        dec_embedding = self.positional_encoding(self.tgt_tok_emb(tgt))
        dec_output = self.decoder(
            x=dec_embedding,
            enc_output=enc_output,
            dec_mask=tgt_mask,
            cross_mask=cross_mask,
        )

        # final classifier
        model_out = self.classifier(dec_output)
        return enc_embedding, enc_output, dec_embedding, dec_output, model_out  # F.log_softmax(model_out, dim=-1) # TODO

    def create_pad_mask(self, q: Tensor, k: Tensor, pad_idx: int) -> Tensor:
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
        return mask.to(self.device)  # shape: (batch_size, 1, q_seq_len, k_seq_len) it can be also broadcasted

    def create_autoregressive_mask(self, q: Tensor, k: Tensor) -> Tensor:  # TODO: src, tgt
        """
        :param q: torch.Tensor, shape: (batch_size, q_seq_len)
        :param k: torch.Tensor, shape: (batch_size, k_seq_len)
        :return: torch.Tensor, shape: (batch_size, q_seq_len, k_seq_len)
        """
        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.ones(len_q, len_k, dtype=torch.bool).tril(diagonal=0)  # TODO: check if triu is correct

        # we can just return here, it will be broadcasted anyway
        # now let's stack it just for shape consistency
        # mask = mask.repeat(batch_size, self.h, 1, 1)
        return mask.to(self.device)

    def create_mask(self, src, tgt):
        # masks
        src_mask = self.create_pad_mask(src, src, self.pad_idx)
        # print("src_mask shape:", src_mask.shape)
        cross_mask = self.create_pad_mask(tgt, src, self.pad_idx)
        # print("cross_mask shape:", cross_mask.shape)

        tgt_pad_mask = self.create_pad_mask(tgt, tgt, self.pad_idx)
        tgt_autoregressive_mask = self.create_autoregressive_mask(tgt, tgt)
        # print("tgt_pad_mask shape:", tgt_pad_mask)
        # print("tgt_autoregressive_mask shape:", tgt_autoregressive_mask)

        tgt_mask = tgt_pad_mask & tgt_autoregressive_mask  # broadcasted TODO: check
        # print("tgt_mask shape:", tgt_mask)

        return src_mask, cross_mask, tgt_pad_mask, tgt_autoregressive_mask, tgt_mask
        # return src_mask, cross_mask, tgt_mask
