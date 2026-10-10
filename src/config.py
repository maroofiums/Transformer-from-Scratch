from dataclasses import dataclass


@dataclass
class TransformerConfig:

    src_vocab_size: int = 100
    tgt_vocab_size: int = 100

    d_model: int = 128
    nums_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512

    dropout: float = 0.1
    max_seq_len: int = 128

    def __post_init__(self):
        if self.d_model % self.nums_heads != 0:
            raise ValueError(
                "d_model must be divisible by nums_heads"
            )

        if self.num_decoder_layers < 1 or self.num_encoder_layers < 1:
            raise ValueError(
                "num_decoder_layers and num_encoder_layers must be greater than 0"
            )

        if self.d_ff < self.d_model:
            raise ValueError(
                "d_ff must be greater than or equal to d_model"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0.0, 1.0)"
            )

            