
from typing import Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# function to collate data samples into batch tesors


class TranslationDataset(Dataset):
    def __init__(self, data, text_transform: Dict, pad_idx, ln_pair: Tuple):
        super().__init__()
        src_ln, trg_ln = ln_pair
        self.pad_idx = pad_idx
        self.tensors = []
        for src, trg in data:
            self.tensors.append(
              (
                text_transform[src_ln](src.rstrip("\n").lower()),
                text_transform[trg_ln](trg.rstrip("\n").lower())
              )   
          )
    def __getitem__(self, idx):
        return self.tensors[idx]
    def __len__(self):
        return len(self.tensors)

    def pad_collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.pad_idx)
        return src_batch, tgt_batch 