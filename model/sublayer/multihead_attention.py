import math
import typing
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int):
        super(MultiHeadAttention, self).__init__()
        # shape vars
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # linear projections
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_concat = nn.Linear(d_model, d_model)

    def forward(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> typing.Tuple[Tensor, Tensor]:
        """
        :param q: torch.Tensor, shape: (batch_size, q_len, d_model)
        :param k: torch.Tensor, shape: (batch_size, k_len, d_model)
        :param v: torch.Tensor, shape: (batch_size, k_len, d_model)
        :param mask: torch.Tensor, shape: (batch_size, h, q_len, k_len)

        :return out: torch.Tensor, shape: (batch_size, q_len, d_model)
        :return attn_score: torch.Tensor, shape: (batch_size, h, q_len, d_model)
        """

        # 1. Linear projection before attention
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)

        # 2. split by the number of heads, shape: (batch_size, h, max_seq_len, d_k)
        q, k, v = self.split(q, self.h), self.split(k, self.h), self.split(v, self.h)

        # 3. reshape tensor (merge_batch) and apply scaled dot product attention
        q, k, v = self.merge_batch(q), self.merge_batch(k), self.merge_batch(v)
        attn_out, attn_score = self.calculate_attn(q, k, v, mask)
        attn_out, attn_score = self.detach_batch(attn_out), self.detach_batch(attn_score)

        # 4. concat attention output
        out = self.concat(attn_out)

        # 5. linear projection after attention
        out = self.fc_concat(out)  # shape: (batch_size, max_seq_len, d_model)

        return out, attn_score

    def split(self, x: Tensor, h: int) -> Tensor:
        """
        split input tensor by the number of head

        :param x : torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :param h : int, the number of heads
        :return: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        """
        batch_size, max_seq_len, d_model = x.size()
        return x.view(batch_size, max_seq_len, h, d_model // h).transpose(1, 2)

    def concat(self, x: Tensor) -> Tensor:
        """
        concat split tensor

        :param x: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        batch_size, h, max_seq_len, d_k = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, max_seq_len, h * d_k)
        )

    def merge_batch(self, x: Tensor) -> Tensor:
        """
        merge the first two dimensions (batch_size and h) into (batch_size * h)
        :param x: 4D Tensor, shape: (batch_size, h, _, _)
        :return: 3D Tensor, shape: (batch_size * h, _, _)
        """
        batch_size, h, dim1, dim2 = x.size()
        x_reshaped = x.contiguous().view(batch_size * h, dim1, dim2)
        return x_reshaped

    def detach_batch(self, x: Tensor) -> Tensor:
        """
        separate the first dimension (batch_size*h) into two (batch_size, h)
        :param x: 3D Tensor, shape: (batch_size * h, _, _)
        :return: 4D Tensor, shape: (batch_size, h, _, _)
        """
        batch_size_times_h, dim1, dim2 = x.size()
        batch_size = batch_size_times_h // self.h
        x_reshaped = x.contiguous().view(batch_size, self.h, dim1, dim2)
        return x_reshaped

    def calculate_attn(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> typing.Tuple[Tensor, Tensor]:
        """
        calculate scaled dot product attention

        :param q: torch.Tensor, shape: (batch_size, h, q_len, d_k)
        :param k: torch.Tensor, shape: (batch_size, h, k_len, d_k)
        :param v: torch.Tensor, shape: (batch_size, h, k_len, d_k)
        :param mask: torch.Tensor, shape: (batch_size, h, q_len, k_len)

        :return attn_out: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        :return attn_score: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        """

        # if mask != None: print("mask", mask.shape) # now (batch_size, seq_len)

        # 1. scaling
        q = q / math.sqrt(self.d_k)

        # 2. QK^T
        attn_score = torch.bmm(q, k.transpose(-2, -1))  # shape: (batch_size * h, q_len, k_len)
        # (3. masking)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, value=float("-inf"))
            pass
        # 4. softmax
        attn_score = F.softmax(attn_score, dim=-1)  # shape: (batch_size * h, q_len, k_len)

        # 5. dot product with V
        attn_out = torch.bmm(attn_score, v)  # shape: (batch_size * h, q_len, d_k)

        return attn_out, attn_score

    def calculate_attn_deprecated(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> typing.Tuple[Tensor, Tensor]:
        """
        calculate scaled dot product attention

        :param q: torch.Tensor, shape: (batch_size, h, q_len, d_k)
        :param k: torch.Tensor, shape: (batch_size, h, k_len, d_k)
        :param v: torch.Tensor, shape: (batch_size, h, k_len, d_k)
        :param mask: torch.Tensor, shape: (batch_size, h, q_len, k_len)

        :return attn_out: torch.Tensor, shape: (batch_size, h, q_len, d_k)
        """
        # 1. QK^T
        attn_score = torch.matmul(
            q, k.transpose(2, 3)
        )  # shape: (batch_size, h, q_len, k_len)

        # 2. scaling
        attn_score = attn_score / math.sqrt(self.d_k)

        # (3. masking)
        if mask is not None:
            # given mask with shape (q_len, k_len),
            # it is broadcasted to (batch_size, h, q_len, k_len)
            # print(attn_score.shape)
            attn_score = attn_score.masked_fill(mask == 0, value=float("-inf"))

        # 4. softmax
        attn_score = attn_score.softmax(dim=-1)  # shape: (batch_size, h, q_len, k_len)

        # 5. dot product with V
        attn_out = torch.matmul(attn_score, v)  # shape: (batch_size, h, q_len, d_k)

        return attn_out, attn_score
