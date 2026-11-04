import torch
import torch.nn as nn

from src.config import TransformerConfig
from src.positional_encoding import PositionalEncoding
from src.encoder import TransformerEncoder
from src.decoder import create_causal_mask, TransformerDecoder


class Transformer(nn.Module):
    """
    Encoder-decoder Transformer.

    Source/target inputs: (batch, sequence_length)
    Decoder output: (batch, sequence_length, target_vocab_size)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.src_embedding = nn.Embedding(
            config.src_vocab_size, config.d_model
        )
        self.tgt_embedding = nn.Embedding(
            config.tgt_vocab_size, config.d_model
        )
        self.embedding_scale = config.d_model ** 0.5

        self.src_positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.tgt_positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        # Keep these module names unchanged for checkpoint compatibility.
        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.output_projection = nn.Linear(
            config.d_model, config.tgt_vocab_size
        )

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode source tokens into memory."""
        x = self.src_embedding(src) * self.embedding_scale
        x = self.src_positional_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode target tokens and return vocabulary logits."""
        x = self.tgt_embedding(tgt) * self.embedding_scale
        x = self.tgt_positional_encoding(x)

        if tgt_mask is None:
            tgt_mask = create_causal_mask(
                tgt.size(-1), device=tgt.device
            )

        x = self.decoder(
            x=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return self.output_projection(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
