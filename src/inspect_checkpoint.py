import torch

checkpoint = torch.load(
    "checkpoints/best_transformer.pt",
    map_location="cpu",
    weights_only=True,
)

print("Best validation loss:", checkpoint["best_val_loss"])
print("Model configuration:")
print(checkpoint["config"])

print("\nSource vocabulary size:", len(checkpoint["src_token_to_id"]))
print("Target vocabulary size:", len(checkpoint["tgt_token_to_id"]))

print("\nSpecial tokens:")
for token in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
    print(
        token,
        "source:", checkpoint["src_token_to_id"].get(token),
        "target:", checkpoint["tgt_token_to_id"].get(token),
    )
