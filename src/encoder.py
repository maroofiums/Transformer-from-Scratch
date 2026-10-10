import torch 
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feed_forward import PositionWiseFeedForward


class TransformerEncoderBlock(nn.Module):

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

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        attention_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=mask
        )

        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout2(feed_forward_output)
        )

        return x

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):

        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout
                )

                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, mask)
        
        return x