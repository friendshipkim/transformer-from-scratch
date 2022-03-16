# TODO: Not necessary

from torch import nn, Tensor

from model.sublayer.token_embedding import TokenEmbedding
from model.sublayer.positional_encoding import PositionalEncoding


class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_seq_len: int):
        super(Embedding, self).__init__()
        self.token_embedding = TokenEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, max_seq_len)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        x = self.token_embedding(x)
        out = self.positional_encoding(x)
        return out



