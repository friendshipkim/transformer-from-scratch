import math
from torch import nn, Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, emb_size: int, vocab_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        """
        :param tokens: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len, emb_size)
        """
        out = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        # multiply embedding weights by sqrt(d_model)
        return out
