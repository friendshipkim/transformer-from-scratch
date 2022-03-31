import math
import typing
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, p_drop: float):
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

        # dropout
        self.dropout = nn.Dropout(p_drop)

        # initialize params
        self._reset_parameters()

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

        # 2. reshape tensor (merge batch and head)
        q, k, v = self.merge_batch(q), self.merge_batch(k), self.merge_batch(v)

        # 3. scaled dot attention
        attn_out, attn_score = self.calculate_attn(q, k, v, mask)
        # attn_out: shape: (batch_size * h, q_len, d_k)
        # attn_score : shape: (batch_size * h, q_len, k_len)

        # 4. reshape tensor (detach batch and head)
        attn_out = self.detach_batch(attn_out)  # shape: (batch_size, q_len, d_model)
        attn_score = self.avg_attn_score(attn_score)  # shape: (batch_size, q_len, k_len)

        # 5. linear projection after attention
        out = self.fc_concat(attn_out)  # shape: (batch_size, q_len, d_model)

        return out, attn_score

    def merge_batch(self, x: Tensor) -> Tensor:
        """
        merge batch and h dimension
        :param x: Tensor, shape: (batch_size, seq_len, d_model)
        :return: Tensor, shape: (batch_size * h, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        return x.transpose(0, 1).contiguous() \
            .view(seq_len, batch_size * self.h, self.d_k) \
            .transpose(0, 1)

    def detach_batch(self, x: Tensor) -> Tensor:
        """
        merge batch and h dimension
        :param x: Tensor, shape: (batch_size * h, seq_len, d_k)
        :return: Tensor, shape: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0) // self.h
        seq_len = x.size(1)
        return x.transpose(0, 1).contiguous() \
            .view(seq_len, batch_size, self.d_model) \
            .transpose(0, 1)

    def avg_attn_score(self, attn_score):
        """
        split batch and h dimension of attn score tensor and average it over head dimension
        :param attn_score: Tensor, shape: (batch_size * h, q_len, k_len)
        :return: Tensor, shape: (batch_size, q_len, k_len)
        """
        _, q_len, k_len = attn_score.size()
        attn_score = attn_score.view(-1, self.h, q_len, k_len)  # shape: (batch_size, h, q_len, k_len)
        avg_attn_score = torch.mean(attn_score, dim=1)

        return avg_attn_score

    def calculate_attn(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None
    ) -> typing.Tuple[Tensor, Tensor]:
        """
        calculate scaled dot product attention

        :param q: torch.Tensor, shape: (batch_size * h, q_len, d_k)
        :param k: torch.Tensor, shape: (batch_size * h, k_len, d_k)
        :param v: torch.Tensor, shape: (batch_size * h, k_len, d_k)
        :param mask: torch.Tensor, shape: (batch_size * h, q_len, k_len)

        :return attn_out: torch.Tensor, shape: (batch_size * h, q_len, d_k)
        :return attn_score: torch.Tensor, shape: (batch_size * h, q_len, k_len)
        """

        # 1. scaling
        q = q / math.sqrt(self.d_k)

        # 2. QK^T
        attn_score = torch.bmm(q, k.transpose(-2, -1))  # shape: (batch_size * h, q_len, k_len)
        # (3. masking)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, value=float("-inf"))

        # 4. softmax
        attn_score = F.softmax(attn_score, dim=-1)  # shape: (batch_size * h, q_len, k_len)

        # (dropout)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attn_score = self.dropout(attn_score)

        # 5. dot product with V
        attn_out = torch.bmm(attn_score, v)  # shape: (batch_size * h, q_len, d_k)

        return attn_out, attn_score

    def _reset_parameters(self):
        # initialize input projection weights
        # check: in/out dimensions affect xavier initialization,
        # initializing with concatenated tensor improves performance
        concat_tensor = torch.empty((3 * self.d_model, self.d_model))
        nn.init.xavier_uniform_(concat_tensor)
        init_tensors = concat_tensor.chunk(3)

        self.fc_q.weight = nn.Parameter(init_tensors[0], requires_grad=True)
        self.fc_k.weight = nn.Parameter(init_tensors[1], requires_grad=True)
        self.fc_v.weight = nn.Parameter(init_tensors[2], requires_grad=True)

        # initialize input / output projection biases as zero
        nn.init.zeros_(self.fc_q.bias)
        nn.init.zeros_(self.fc_k.bias)
        nn.init.zeros_(self.fc_v.bias)
        nn.init.zeros_(self.fc_concat.bias)
