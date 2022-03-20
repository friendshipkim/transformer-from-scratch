from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # print("token_embedding input shape:", token_embedding.shape)
        # print("positional encoding tensor shape:", self.pos_embedding.shape)
        # print("seqlen:", token_embedding.size(1))
        # print("summed encoding tensor shape:", self.pos_embedding[:, :token_embedding.size(1), :].shape)

        # batch_first input: (batch_size, seq_len, emb_size)
        # out = self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        out = self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])
        # print("embedding output shape:", out.shape)
        # print()
        return out


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class BaselineTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int,
                 dropout: float,
                 pad_idx: int,
                 device: str):
        super(BaselineTransformer, self).__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)  # Added
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor, ):
        # masking
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, trg)
        # print("src_mask shape:", src_mask.shape)
        # print("tgt_mask shape:", tgt_mask.shape)
        # print("src_padding_mask shape:", src_padding_mask.shape)
        # print("tgt_padding_mask shape", tgt_padding_mask.shape)

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory, outs = self.transformer(src_emb, tgt_emb,
                                        src_mask=src_mask,  # additive mask for the src sequence, shape: (S, S)
                                        tgt_mask=tgt_mask,  # additive mask for the tgt sequence, shape: (T, T)
                                        memory_mask=None,  # additive mask for the encoder output, shape: (T, S)
                                        src_key_padding_mask=src_padding_mask,  # mask for src keys per batch, shape: (N, S)
                                        tgt_key_padding_mask=tgt_padding_mask,  # mask for tgt keys per batch, shape: (N, T)
                                        memory_key_padding_mask=src_padding_mask)  # mask for memory keys per batch, shape: (N, S)
        return src_emb, tgt_emb, memory, outs, self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):  # changed to batch_first
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
