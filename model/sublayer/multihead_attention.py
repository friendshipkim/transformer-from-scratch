import math
import typing
import torch
from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_concat = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> typing.Dict[str, Tensor]:
        # TODO: mask_future necessary?
        """
        :param q: torch.Tensor, shape: (batch_size, q_max_seq_len, d_model)
        :param k: torch.Tensor, shape: (batch_size, k_max_seq_len, d_model)
        :param v: torch.Tensor, shape: (batch_size, v_max_seq_len, d_model)
        :param mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)

        :return: torch.Tensor, shape: (batch_size, q_max_seq_len, d_model)
        """

        # 1. Linear projection before attention
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)

        # 2. split by the number of heads, shape: (batch_size, h, max_seq_len, d_k)
        q, k, v = self.split(q, self.h), self.split(k, self.h), self.split(v, self.h)

        # 3. apply scaled dot product attention
        attn_dict = self.calculate_attn(q, k, v, mask)
        out = attn_dict["output"]
        attn_score = attn_dict["attn_score"]

        # 4. concat attention output
        out = self.concat(out)

        # 5. linear projection after attention
        out = self.fc_concat(out)  # shape: (batch_size, max_seq_len, d_model)

        return {"output": out,
                "attn_score": attn_score}

    def split(self, tensor: Tensor, h: int) -> Tensor:
        """
        split input tensor by the number of head

        :param: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :return: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        """
        batch_size, max_seq_len, d_model = tensor.size()
        return tensor.view(batch_size, max_seq_len, h, d_model // h).transpose(1, 2)

    def concat(self, tensor: Tensor) -> Tensor:
        """
        concat split tensor

        :param: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        batch_size, h, max_seq_len, d_k = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, max_seq_len, h*d_k)

    def calculate_attn(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> typing.Dict[str, Tensor]:
        """
        calculate scaled dot product attention

        :param q: torch.Tensor, shape: (batch_size, h, q_max_seq_len, d_k)
        :param k: torch.Tensor, shape: (batch_size, h, k_max_seq_len, d_k)
        :param v: torch.Tensor, shape: (batch_size, h, v_max_seq_len, d_k)
        :param mask: torch.Tensor, shape: (batch_size, h, q_max_seq_len, k_max_seq_len)

        :return: torch.Tensor, shape: (batch_size, h, max_seq_len, d_k)
        """

        # 1. QK^T
        attn_score = torch.matmul(q, k.transpose(2, 3))  # shape: (batch_size, h, q_max_seq_length, k_max_seq_length)

        # 2. scaling
        attn_score = attn_score / math.sqrt(self.d_k)

        # (3. masking)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, value=float("-Inf"))

        # 4. softmax
        attn_score = attn_score.softmax(dim=3)  # shape: (batch_size, h, q_max_seq_length, k_max_seq_length)

        # 5. dot product with V
        out = torch.matmul(attn_score, v)  # shape: (batch_size, h, max_seq_length, d_k)

        out_dict = {"output": out,
                    "attn_score": attn_score}
        return out_dict



