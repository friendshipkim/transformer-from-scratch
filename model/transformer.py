import config
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.layer.embedding import Embedding
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
            vocab_size: int,
            max_seq_len: int,
            pad_idx: int,  # TODO: check if source, target use same pad_idx
            device: str
    ):
        super(Transformer, self).__init__()

        self.pad_idx = pad_idx
        self.h = h  # TODO: remove to use broadcasting

        # encoder
        self.enc_embedding = Embedding(
            d_model=d_model, vocab_size=vocab_size, max_seq_len=max_seq_len
        )
        self.encoder = Encoder(
            n_layers=n_layers,
            d_model=d_model,
            h=h,
            ffn_hidden=ffn_hidden,
            p_drop=p_drop,
        )

        # decoder
        self.dec_embedding = self.enc_embedding
        # TODO: how to make shared embedding, what if source/target vocab size is different?
        # self.dec_embedding = Embedding(d_model=d_model, vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.decoder = Decoder(
            n_layers=n_layers,
            d_model=d_model,
            h=h,
            ffn_hidden=ffn_hidden,
            p_drop=p_drop,
        )

        # final classifier
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, enc_input: Tensor, dec_input: Tensor) -> Tensor:
        """
        :param enc_input: torch.Tensor, shape: (batch_size, max_seq_len)
        :param dec_input: torch.Tensor, shape: (batch_size, max_seq_len)

        :return: torch.Tensor, shape: (batch_size, max_seq_len, vocab_size)
        """

        # masks
        enc_mask = self.make_pad_mask(enc_input, enc_input, self.pad_idx)
        print("enc_mask shape:", enc_mask.shape)
        cross_mask = self.make_pad_mask(dec_input, enc_input, self.pad_idx)
        print("cross_mask shape:", cross_mask.shape)
        dec_mask = self.make_pad_mask(dec_input, dec_input, self.pad_idx) & \
                   self.make_autoregressive_mask(dec_input, dec_input)  # broadcasted TODO: check
        print("dec_mask shape:", dec_mask.shape)

        # encoder
        enc_embedding = self.enc_embedding(x=enc_input)
        enc_output = self.encoder(x=enc_embedding, enc_mask=enc_mask)  # context

        # decoder
        dec_embedding = self.dec_embedding(x=dec_input)
        dec_out = self.decoder(
            x=dec_embedding,
            enc_output=enc_output,
            dec_mask=dec_mask,
            cross_mask=cross_mask,
        )

        # final classifier
        model_out = self.classifier(dec_out)
        return model_out  # F.log_softmax(model_out, dim=-1) # TODO

    def make_pad_mask(self, q: Tensor, k: Tensor, pad_idx: int) -> Tensor:
        """
        Create a mask to hide padding

        :param q: torch.Tensor, shape: (batch_size, q_max_seq_len)
        :param k: torch.Tensor, shape: (batch_size, k_max_seq_len)
        :param pad_idx: int, pad token index

        :return: torch.Tensor, shape: (batch_size, 1, q_max_seq_len, k_max_seq_len)
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
        return mask  # TODO: it can be also broadcasted

    def make_autoregressive_mask(self, q: Tensor, k: Tensor) -> Tensor:
        """
        :param q: torch.Tensor, shape: (batch_size, q_max_seq_len)
        :param k: torch.Tensor, shape: (batch_size, k_max_seq_len)
        :return: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)
        """
        batch_size = q.size(0)
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.ones(len_q, len_k, dtype=torch.bool).tril(diagonal=0)

        # we can just return here, it will be broadcasted anyway
        # now let's stack it just for shape consistency
        mask = mask.repeat(batch_size, self.h, 1, 1)
        return mask
