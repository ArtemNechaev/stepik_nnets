
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# function to collate data samples into batch tesors
def pad_collate_fn(batch, PAD_IDX ):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

class TranslationDataset(Dataset):
    def __init__(self, data, text_transform: Dict, src_ln: str, trg_ln:str):
        super().__init__()
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