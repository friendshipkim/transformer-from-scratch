import torch
import math
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 p_drop: float,
                 maxlen: int = 5000):  # assume max seq len = 5000
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(maxlen, emb_size)
        position = torch.arange(0,  maxlen)[
            ..., None
        ]  # [max_seq_len, 1], use [..., None] instead of .unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size)
        )  # log space
        # div_term = torch.reciprocal(torch.pow(10000, torch.arange(0, emb_size, 2) / emb_size))
        encoding[:, 0::2] = torch.sin(
            position * div_term
        )  # position * div_term broadcasted to [max_seq_len, emb_size/2]
        encoding[:, 1::2] = torch.cos(position * div_term)

        # my implementation
        # position = torch.arange(0, max_seq_len)[..., None]
        # sinusoid_input = position / (10000 ** torch.arange(0, emb_size, 2) / emb_size)
        # encoding[:, 0::2] = torch.sin(sinusoid_input)
        # encoding[:, 1::2] = torch.cos(sinusoid_input)

        # self.encoding = torch.nn.Parameter(encoding)
        # self.encoding.require_grad = False

        # pytorch tutorial implementation
        self.dropout = nn.Dropout(p_drop)  # TODO: dropout necessary for embedding??

        encoding = encoding[None, :]  # add dimension to match tutorial
        self.register_buffer('encoding', encoding)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, seq_len, emb_size)
        :return: torch.Tensor, shape: (batch_size, seq_len, emb_size)
        """
        # print("token_embedding input shape:", x.shape)
        # print("encoding shape:", self.encoding.shape)
        # print("summed encoding shape:", self.encoding[:, :x.size(1), :].shape)
        out = self.dropout(x + self.encoding[:, :x.size(1), :])
        # print("out shape:", out.shape)
        # print()
        return out
