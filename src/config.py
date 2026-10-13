from dataclasses import dataclass
import random

from src.tokenizer import SimpleTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 16
MAX_SEQ_LEN = 64
TRAIN_LIMIT = 5000
VAL_LIMIT = 500
EPOCHS = 5
LEARNING_RATE = 3e-4

PAD = SimpleTokenizer.PAD

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

@dataclass
class TransformerConfig:

    src_vocab_size: int = 100
    tgt_vocab_size: int = 100

    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512

    dropout: float = 0.1
    max_seq_len: int = 128

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
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

            