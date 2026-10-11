import torch
from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(    
    src_pad_id,
    tgt_pad_id  
):
    def collate_fn(batch):

        sources = [item["src"] for item in batch]
        targets = [item["tgt"] for item in batch]

        src = pad_sequence(
            sources,
            batch_first=True,
            padding_value=src_pad_id
        )

        tgt = pad_sequence(
            targets,
            batch_first=True,
            padding_value=tgt_pad_id
        )

        return {
            "src": src,
            "tgt": tgt,
            "source_text": [
                item["source_text"] for item in batch
            ],
            "target_text": [
                item["target_text"] for item in batch
            ]
        }

    return collate_fn