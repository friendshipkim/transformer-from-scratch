import torch
import math
from torch import nn, Tensor


class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_seq_len: int):
        super(Embedding, self).__init__()
        self.token_embedding = TokenEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, max_seq_len)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        x = self.token_embedding(x)
        out = self.positional_encoding(x)
        return out


class TokenEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, max_seq_len)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        out = self.embedding(x) * math.sqrt(
            self.d_model
        )  # multiply embedding weights by sqrt(d_model)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len)[
            ..., None
        ]  # [max_seq_len, 1], use [..., None] instead of .unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # log space
        # div_term = torch.reciprocal(torch.pow(10000, torch.arange(0, d_model, 2) / d_model))
        encoding[:, 0::2] = torch.sin(
            position * div_term
        )  # position * div_term broadcasted to [max_seq_len, d_model/2]
        encoding[:, 1::2] = torch.cos(position * div_term)

        # my implementation
        # position = torch.arange(0, max_seq_len)[..., None]
        # sinusoid_input = position / (10000 ** torch.arange(0, d_model, 2) / d_model)
        # encoding[:, 0::2] = torch.sin(sinusoid_input)
        # encoding[:, 1::2] = torch.cos(sinusoid_input)

        self.encoding = encoding

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        seq_len = x.size(1)
        # TODO: why do we need seq_length besides max_seq_length?
        # TODO: do we have to use torch.autograd.Variable?, self.encoding already requires_grad=False
        pe = torch.autograd.Variable(self.encoding[:seq_len, :], requires_grad=False)
        x = x + pe
        return x  # x = x + self.encoding[:seq_len, :]
