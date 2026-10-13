import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset_split,
        src_tokenizer: SimpleTokenizer,
        tgt_tokenizer: SimpleTokenizer,
        max_seq_len: int
    ):

        self.dataset_split = dataset_split
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len


    def __len__(self):

        return len(self.dataset_split)

    def __getitem__(self, index):

        example = self.dataset_split[index]["translation"]

        source_text = example["en"]
        target_text = example["fr"]

        source_ids = self.src_tokenizer.encode(source_text)
        target_ids = self.tgt_tokenizer.encode(target_text)

        source_ids = source_ids[:self.max_seq_len]
        target_ids = target_ids[:self.max_seq_len]

        source_ids[-1] = self.src_tokenizer.token_to_id[
            self.src_tokenizer.EOS
        ]

        target_ids[-1] = self.tgt_tokenizer.token_to_id[
            self.tgt_tokenizer.EOS
        ]

        return {
            "src": torch.tensor(source_ids, dtype=torch.long),
            "tgt": torch.tensor(target_ids, dtype=torch.long),
            "source_text": source_text,
            "target_text": target_text
        }