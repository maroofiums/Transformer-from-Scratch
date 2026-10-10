import math

import torch
import torch.nn as nn

from typing import Tuple

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the scaled dot product attention.

    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
        k: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
        v: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v)
        mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
        dropout: Optional dropout layer

    Returns:
        output: Tensor of shape (batch_size, num_heads, seq_len_q, d_v)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(
            mask,
            float('-inf')
        )

    weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, v)

    return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape

        x = x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )

        return x.transpose(1, 2)  


    def combine_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine the heads into a single tensor.

        Args:
            x: Tensor of shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.reshape(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)

        Returns:
            output: Tensor of shape (batch_size, seq_len_q, d_model)
            attention_weights: Tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        output, attention_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
            self.dropout
        )

        output = self.combine_heads(output)

        output = self.w_o(output)

        return output, attention_weights