import typing

import torch
from torch import nn, Tensor

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

    def forward(self, src: Tensor, tgt: Tensor) -> typing.Tuple:
        """
        :param src: torch.Tensor, shape: (batch_size, src_seq_len)
        :param tgt: torch.Tensor, shape: (batch_size, tgt_seq_len)

        :return: torch.Tensor, shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # mask
        src_pad_mask, tgt_pad_mask, tgt_autoregressive_mask = self.create_mask(src, tgt)

        # encoder
        enc_embedding = self.positional_encoding(self.src_tok_emb(src))
        enc_output = self.encoder(x=enc_embedding,
                                  src_pad_mask=src_pad_mask)  # memory

        # decoder
        dec_embedding = self.positional_encoding(self.tgt_tok_emb(tgt))
        dec_output = self.decoder(
            x=dec_embedding,
            enc_output=enc_output,
            tgt_pad_mask=tgt_pad_mask,
            tgt_autoregressive_mask=tgt_autoregressive_mask,
            memory_pad_mask=src_pad_mask
        )

        # final classifier
        model_out = self.classifier(dec_output)
        return enc_embedding, enc_output, dec_embedding, dec_output, model_out  # F.log_softmax(model_out, dim=-1) # TODO

    def create_pad_mask(self, x: Tensor) -> Tensor:
        """
        Create a mask to hide padding

        :param x: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len)
        """
        return x != self.pad_idx

    def create_autoregressive_mask(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len, seq_len)
        """
        seq_len = x.size(1)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).tril(diagonal=0)
        return mask

    def create_mask(self, src: Tensor, tgt: Tensor) -> typing.Tuple[Tensor, Tensor, Tensor]:
        # masks
        src_pad_mask = self.create_pad_mask(src)
        tgt_pad_mask = self.create_pad_mask(tgt)
        tgt_autoregressive_mask = self.create_autoregressive_mask(tgt)
        return src_pad_mask, tgt_pad_mask, tgt_autoregressive_mask

# for deprecated
# .unsqueeze(-2).unsqueeze(1)