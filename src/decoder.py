import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feed_forward import PositionWiseFeedForward


def create_causal_mask(
    seq_len: int,
    device: torch.device | None = None
) -> torch.Tensor:

    """
    Returns a Boolean mask of shape (seq_len, seq_len).

    True = block attention
    False = allow attention
    """

    return torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=device
        ),
        diagonal=1
    )


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float
    ):
        
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.dropout3  = nn.Dropout(dropout)


    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask
        )

        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        cross_attention_output, _ = self.cross_attention(
            query=x,
            key=memory,
            value=memory,
            mask=memory_mask
        )

        x = self.norm2(
            x + self.dropout2(cross_attention_output)
        )

        feed_forward_output = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout3(feed_forward_output)
        )

        return x

        