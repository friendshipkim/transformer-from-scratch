import torch
from torch import nn

from model.layer.embedding import Embedding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, n_layers, d_model, h, ffn_hidden, p_drop, vocab_size, d_embed, vocab):
        super(Transformer, self).__init__()
        # encoder
        self.enc_embedding = Embedding(d_embed, vocab)
        self.encoder = Encoder(n_layers, d_model, h, ffn_hidden, p_drop)

        # decoder
        self.enc_embedding = Embedding(d_embed, vocab)
        self.decoder = Decoder(n_layers, d_model, h, ffn_hidden, p_drop)

        # linear classifier
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # masks
        src_mask = self.make_pad_mask(src, src)
        cross_mask = self.make_pad_mask(tgt, src)
        tgt_mask = self.make_pad_mask(tgt, tgt) * self.make_no_peak_mask(tgt, tgt)

        enc_embedding = self.enc_embedding(src)
        enc_out = self.encoder(enc_embedding, src_mask)  # context

        dec_embedding = self.dec_embedding(tgt)
        dec_out = self.decoder(dec_embedding, enc_out, tgt_mask, cross_mask)

        model_out = self.classifier(dec_out)  # final classification layer
        return model_out

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
