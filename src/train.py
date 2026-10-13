# File: src/train.py

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


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 16
MAX_SEQ_LEN = 64

TRAIN_LIMIT = 20_000
VAL_LIMIT = 1_000

EPOCHS = 50
PATIENCE = 7
MIN_DELTA = 0.01

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

PAD = SimpleTokenizer.PAD

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Reproducibility; may reduce performance slightly.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Tokenizers
# ============================================================

def build_tokenizers(train_split):
    """Build separate English and French vocabularies."""

    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    src_tokenizer.train(
        (
            row["translation"]["en"]
            for row in train_split
        ),
        min_frequency=2,
    )

    tgt_tokenizer.train(
        (
            row["translation"]["fr"]
            for row in train_split
        ),
        min_frequency=2,
    )

    return src_tokenizer, tgt_tokenizer


# ============================================================
# 3. Attention masks
# ============================================================

def make_attention_masks(
    src,
    tgt_input,
    src_pad_id,
    tgt_pad_id,
):
    """
    True means blocked in our attention implementation.

    Source mask: blocks source padding keys.
    Target mask: blocks target padding keys and future tokens.
    """

    # Shape: (B, 1, 1, S)
    src_padding_mask = (
        src == src_pad_id
    )[:, None, None, :]

    # Shape: (B, 1, 1, T)
    tgt_padding_mask = (
        tgt_input == tgt_pad_id
    )[:, None, None, :]

    target_length = tgt_input.size(1)

    # Shape: (T, T)
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


# ============================================================
# 4. One training or validation epoch
# ============================================================

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
        # Decoder input: BOS, token_1, ..., token_(n-1)
        # Labels:        token_1, ..., token_n, EOS
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

                # Prevent excessively large gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

        # Ignore padding tokens when averaging epoch loss.
        valid_tokens = (
            labels != tgt_pad_id
        ).sum().item()

        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


# ============================================================
# 5. Main training pipeline
# ============================================================

def main():
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load and split dataset
    # --------------------------------------------------------

    dataset = load_translation_dataset()
    all_data = dataset["train"]

    splits = all_data.train_test_split(
        test_size=0.1,
        seed=SEED,
    )

    # Shuffle before selecting subsets so we don't simply take
    # the first portion of a potentially ordered book dataset.
    train_split = splits["train"].shuffle(
        seed=SEED
    )

    val_split = splits["test"].shuffle(
        seed=SEED
    )

    train_split = train_split.select(
        range(min(TRAIN_LIMIT, len(train_split)))
    )

    val_split = val_split.select(
        range(min(VAL_LIMIT, len(val_split)))
    )

    print("Training examples:", len(train_split))
    print("Validation examples:", len(val_split))

    # --------------------------------------------------------
    # Build vocabularies using training data only
    # --------------------------------------------------------

    src_tokenizer, tgt_tokenizer = build_tokenizers(
        train_split
    )

    print(
        "Source vocabulary:",
        src_tokenizer.vocab_size,
    )

    print(
        "Target vocabulary:",
        tgt_tokenizer.vocab_size,
    )

    # --------------------------------------------------------
    # Create PyTorch datasets
    # --------------------------------------------------------

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

    collate_fn = make_collate_fn(
        src_pad_id,
        tgt_pad_id,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    config = TransformerConfig(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    )

    model = Transformer(config).to(DEVICE)

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )

    # Padding labels do not contribute to the loss.
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_pad_id,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Checkpoint and early-stopping configuration
    # --------------------------------------------------------

    output_dir = Path("checkpoints")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir / "best_transformer.pt"
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\nStarting training...\n")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            src_pad_id=src_pad_id,
            tgt_pad_id=tgt_pad_id,
            training=True,
        )

        val_loss = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            src_pad_id=src_pad_id,
            tgt_pad_id=tgt_pad_id,
            training=False,
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {val_loss:.4f}"
        )

        # Save every new best checkpoint.
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "config": config.__dict__,
                    "src_token_to_id": (
                        src_tokenizer.token_to_id
                    ),
                    "tgt_token_to_id": (
                        tgt_tokenizer.token_to_id
                    ),
                    "src_pad_id": src_pad_id,
                    "tgt_pad_id": tgt_pad_id,
                    "best_val_loss": best_val_loss,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )

            print(
                f"Saved best checkpoint "
                f"(epoch {epoch}, "
                f"validation loss {best_val_loss:.4f})."
            )

            if improvement >= MIN_DELTA:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        else:
            epochs_without_improvement += 1

        # Stop when validation performance stops improving.
        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}."
            )
            print(
                f"Best validation loss: "
                f"{best_val_loss:.4f}"
            )
            break

    print("\nTraining complete.")
    print("Best validation loss:", best_val_loss)
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()