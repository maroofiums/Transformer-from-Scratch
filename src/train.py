
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TransformerConfig
from src.data import load_translation_dataset
from src.tokenizer import SimpleTokenizer
from src.translation_dataset import TranslationDataset
from src.collate import make_collate_fn
from src.transformer import Transformer


# -------------------------
# Configuration
# -------------------------

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tokenizers(train_split):
    """Build separate source and target vocabularies."""

    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    src_tokenizer.train(
        (row["translation"]["en"] for row in train_split),
        min_frequency=2,
    )

    tgt_tokenizer.train(
        (row["translation"]["fr"] for row in train_split),
        min_frequency=2,
    )

    return src_tokenizer, tgt_tokenizer


def make_attention_masks(src, tgt_input, src_pad_id, tgt_pad_id):
    """
    True means blocked in our MultiHeadAttention implementation.
    """

    # (B, 1, 1, S): block source padding keys.
w

    target_length = tgt_input.size(1)

    # True above the diagonal blocks future positions.
    causal_mask = torch.triu(
        torch.ones(
            target_length,
            target_length,
            dtype=torch.bool,
            device=tgt_input.device,
        ),
        diagonal=1,
    )

    # Broadcast to (B, 1, T, T).
    tgt_mask = (
        tgt_padding_mask
        | causal_mask[None, None, :, :]
    )

    return src_padding_mask, tgt_mask


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    src_pad_id,
    tgt_pad_id,
    training,
):
    model.train(training)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        # Teacher forcing:
        # decoder input: BOS ... previous token
        # labels:        next token ... EOS
        tgt_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        src_mask, tgt_mask = make_attention_masks(
            src,
            tgt_input,
            src_pad_id,
            tgt_pad_id,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(
                src=src,
                tgt=tgt_input,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=src_mask,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            if training:
                loss.backward()

                # Helps prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        # Average by non-padding target tokens, not batches.

       valid_tokens = labels.ne(tgt_pad_id).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


def main():
    print("Device:", DEVICE)

    dataset = load_translation_dataset()
    all_train = dataset["train"]

    # Split first; vocabulary is learned only from training data.
    splits = all_train.train_test_split(
        test_size=0.1,
        seed=SEED,
    )

    train_split = splits["train"]
    val_split = splits["test"]

    # Keep the first run manageable.
    train_split = train_split.select(
        range(min(TRAIN_LIMIT, len(train_split)))
    )
    val_split = val_split.select(
        range(min(VAL_LIMIT, len(val_split)))
    )

    print("Training examples:", len(train_split))
    print("Validation examples:", len(val_split))

    src_tokenizer, tgt_tokenizer = build_tokenizers(train_split)

    print("Source vocabulary:", src_tokenizer.vocab_size)
    print("Target vocabulary:", tgt_tokenizer.vocab_size)

    train_data = TranslationDataset(
        train_split,
        src_tokenizer,
        tgt_tokenizer,
        max_seq_len=MAX_SEQ_LEN,
    )

    val_data = TranslationDataset(
        val_split,
        src_tokenizer,
        tgt_tokenizer,
        max_seq_len=MAX_SEQ_LEN,
    )

    src_pad_id = src_tokenizer.token_to_id[PAD]
    tgt_pad_id = tgt_tokenizer.token_to_id[PAD]

    collate_fn = make_collate_fn(src_pad_id, tgt_pad_id)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    config = TransformerConfig(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
        max_seq_len=MAX_SEQ_LEN,
    )

    model = Transformer(config).to(DEVICE)

    # Padding labels do not contribute to the loss.
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_pad_id,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            src_pad_id,
            tgt_pad_id,
            training=True,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            src_pad_id,
            tgt_pad_id,
            training=False,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "src_token_to_id": src_tokenizer.token_to_id,
                    "tgt_token_to_id": tgt_tokenizer.token_to_id,
                    "src_pad_id": src_pad_id,
                    "tgt_pad_id": tgt_pad_id,
                    "best_val_loss": best_val_loss,
                },
                output_dir / "best_transformer.pt",
            )

            print("Saved best checkpoint.")

    print("Training complete.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    main()
