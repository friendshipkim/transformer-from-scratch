import torch
from torch import nn

from model.layer.embedding import Embedding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, h: int, ffn_hidden: int, p_drop: float, vocab_size: int, max_seq_len: int):
        super(Transformer, self).__init__()
        # encoder
        self.enc_embedding = Embedding(d_model=d_model, vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.encoder = Encoder(n_layers=n_layers, d_model=d_model, h=h, ffn_hidden=ffn_hidden, p_drop=p_drop)

        # decoder
        self.dec_embedding = self.enc_embedding
        # TODO: how to make shared embedding
        # self.dec_embedding = Embedding(d_model=d_model, vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.decoder = Decoder(n_layers=n_layers, d_model=d_model, h=h, ffn_hidden=ffn_hidden, p_drop=p_drop)

        # final classifier
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, enc_input, dec_input):
        # masks TODO
        enc_mask = self.make_pad_mask(enc_input, enc_input)
        cross_mask = self.make_pad_mask(dec_input, enc_input)
        dec_mask = self.make_pad_mask(dec_input, dec_input) * self.make_no_peak_mask(dec_input, dec_input)

        # encoder
        enc_embedding = self.enc_embedding(x=enc_input)
        enc_output = self.encoder(x=enc_embedding, enc_mask=enc_mask)  # context

        # decoder
        dec_embedding = self.dec_embedding(x=dec_input)
        dec_out = self.decoder(x=dec_embedding, enc_output=enc_output, dec_mask=dec_mask, cross_mask=cross_mask)

        # final classifier
        model_out = self.classifier(dec_out)
        return model_out

    # masking

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # batch_size x 1 x 1 x len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q x len_k
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask
