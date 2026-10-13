
import torch

from src.config import TransformerConfig
from src.tokenizer import SimpleTokenizer
from src.transformer import Transformer


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "checkpoints/best_transformer.pt"


def restore_tokenizer(token_to_id):
    tokenizer = SimpleTokenizer()
    tokenizer.token_to_id = token_to_id
    tokenizer.id_to_token = {
        idx: token for token, idx in token_to_id.items()
    }
    return tokenizer


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    config = TransformerConfig(**checkpoint["config"])
    model = Transformer(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    src_tokenizer = restore_tokenizer(
        checkpoint["src_token_to_id"]
    )
    tgt_tokenizer = restore_tokenizer(
        checkpoint["tgt_token_to_id"]
    )

    return model, src_tokenizer, tgt_tokenizer, checkpoint


@torch.inference_mode()
def translate(
    model,
    src_tokenizer,
    tgt_tokenizer,
    sentence,
    max_new_tokens=64,
):
    src_ids = src_tokenizer.encode(sentence)

    max_len = model.config.max_seq_len
    src_ids = src_ids[:max_len]

    # Preserve EOS if the input was truncated.
    src_ids[-1] = src_tokenizer.token_to_id[
        SimpleTokenizer.EOS
    ]

    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    src_pad_id = src_tokenizer.token_to_id[
        SimpleTokenizer.PAD
    ]

    src_mask = (src == src_pad_id)[:, None, None, :]

    memory = model.encode(
        src,
        src_mask=src_mask,
    )

    bos_id = tgt_tokenizer.token_to_id[
        SimpleTokenizer.BOS
    ]
    eos_id = tgt_tokenizer.token_to_id[
        SimpleTokenizer.EOS
    ]

    generated = [bos_id]

    for _ in range(min(max_new_tokens, max_len - 1)):
        tgt = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE,
        )

        logits = model.decode(
            tgt=tgt,
            memory=memory,
            memory_mask=src_mask,
        )

        next_token = logits[0, -1].argmax().item()
        generated.append(next_token)

        if next_token == eos_id:
            break

    return tgt_tokenizer.decode(generated)


def main():
    model, src_tok, tgt_tok, checkpoint = load_model()

    print("Device:", DEVICE)
    print("Best validation loss:", checkpoint["best_val_loss"])
    print("Type 'quit' to exit.")

    while True:
        sentence = input("\nEnglish: ").strip()

        if sentence.lower() == "quit":
            break

        if not sentence:
            continue

        result = translate(
            model,
            src_tok,
            tgt_tok,
            sentence,
        )

        print("French:", result)


if __name__ == "__main__":
    main()
