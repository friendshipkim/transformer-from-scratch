import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        # 1. Linear projection before attention
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)

        # 2. split by the number of heads
        q, k, v = self.split(q, self.h), self.split(k, self.h), self.split(v, self.h)

        # 3. apply scaled dot product attention
        out = self.calculate_attn(q, k, v, mask)

        # 4. concat attention output
        out = self.concat(out)

        # 5. linear projection after attention
        out = self.fc_concat(out)  # shape: (batch_size, max_seq_len, d_model)

        return out

    def split(self, tensor, h):
        """
        split input tensor by the number of head

        :param: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :return: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        """
        batch_size, max_seq_len, d_model = tensor.size()
        return tensor.view(batch_size, max_seq_len, h, d_model // h).transpose(1, 2)

    def concat(self, tensor):
        """
        concat split tensor

        :param: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        batch_size, h, max_seq_len, d_k = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, max_seq_len, h*d_k) # contiguous()?

    def calculate_attn(self, q, k, v, mask):
        """
        calculate scaled dot product attention

        :param: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        :return: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        """

        # 1. QK^T
        attn_score = torch.matmul(q, k.transpose(2, 3))  # shape: (batch_size, max_seq_length, max_seq_length)

        # 2. scaling
        attn_score = attn_score / math.sqrt(self.d_k)

        # (3. masking)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, value=float("-Inf"))

        # 4. softmax
        attn_score = attn_score.softmax(dim=3)

        # 5. dot product with V
        out = torch.matmul(attn_score, v)  # shape: (batch_size, h, max_seq_length, d_k)

        return out
