import torch
from typing import List
from torchtext.vocab import  FastText

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices

class TensorTransform(object):
    def __init__(self, bos_idx, eos_idx) -> None:
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
    def __call__(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.bos_idx], dtype=torch.long),
                      torch.tensor(token_ids, dtype=torch.long),
                      torch.tensor([self.eos_idx], dtype=torch.long)))

class myFastText(FastText):
    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            vector = torch.Tensor(1, self.dim).zero_()
            num_vectors = 0
            chars = list(token)
            for n in [3,4,5]:
                end = len(chars) - n + 1
                grams = [chars[i:(i + n)] for i in range(end)]
                for gram in grams:
                    gram_key = ''.join(gram)
                    if gram_key in self.stoi:
                        vector += self.vectors[self.stoi[gram_key]]
                        num_vectors += 1
            if num_vectors > 0:
                vector /= num_vectors
            else:
                vector = self.unk_init(vector)
            return vector

    def __call__(self, corpus: List[List[str]] ):
        res = torch


