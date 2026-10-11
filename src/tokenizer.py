import re
from collections import counter


class SimpleTokenizer:
    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"

    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def  __init___(self) -> None:
        self.token_to_id = {
            token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)
        }
        self.id_to_token = {
            idx: token for token, idx in self.token_to_id.items()
        }

    def tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


    def train(
        self,
        sentences: list[str],
        min_freq: int = 2,
    ):
        counter = Counter()

        for sentence in sentences:
            counter.update(self.tokenize(sentence))

        for token, freq in counter.most_common():
            if freq < min_freq:
                continue

            if token not in self.token_to_id:
                token_id = len(self.token_to_id)

                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:

        tokens = self.tokenize(text)

        ids = [
            self.token_to_id.get(token, self.token_to_id[self.UNK])
            for token in tokens
        ]

        if add_special_tokens:
            ids =[
                self.token_to_id[self.BOS],
                *ids,
                self.token_to_id[self.EOS],
            ]

        return ids

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:

        tokens = []
        
        special_ids = {
            self.token_to_id[token]
            for token in self.SPECIAL_TOKENS
        }

        for token_id in ids:
            token = self.id_to_token.get(
                int(token_id),
                self.UNK
            )

            if skip_special_tokens and token in self.SPECIAL_TOKENS:
                continue

            tokens.append(token)

        text = " ".join(tokens)

        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text