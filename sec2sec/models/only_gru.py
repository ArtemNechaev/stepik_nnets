from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class OnlyGRU(nn.Module):
    def __init__(self, input_dim, output_dim, emd_size, hidden_size, num_layers,  pad_idx, eos_idx, device, dropout=0.3, src_embed=None, trg_embed=None) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emd_size = emd_size
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.device = device

        if src_embed != None:
            self.src_embedding = src_embed
        else:
            self.src_embedding = nn.Embedding(
            input_dim, emd_size, padding_idx=pad_idx)

        if trg_embed != None:
            self.trg_embedding = trg_embed
        else:
            self.trg_embedding = nn.Embedding(
                output_dim, emd_size, padding_idx=pad_idx)
        
        self.encoder = nn.GRU(emd_size, hidden_size, num_layers, bidirectional=True)
        self.decoder = nn.GRU(emd_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.enc2dec = nn.Linear(2*self.encoder.num_layers, self.decoder.num_layers)
        self.to_trg_vocab_size = nn.Linear(hidden_size * 2, output_dim)

        self.forward_mode = 'next_word'

    def create_mask(self, src, trg):
        src_mask = (src != self.pad_idx).permute(1,0).unsqueeze(1)
        trg_mask = (trg != self.pad_idx).permute(1,0).unsqueeze(2)
        mask = trg_mask * src_mask

        return {'mask': mask}

    @property
    def forward_mode(self):
        return self._forward_mode

    @forward_mode.setter
    def forward_mode(self, value):
        assert(value in ['next_word', 'greedy'])
        self._forward_mode = value


    def forward(self, src, trg, teacher_forcing_ratio, *args):
        """
        src - src len x batch_size
        trg - trg sent len x batch_size


        return:
            outputs - trg sent len x batch_size x trg_vocab_size (output_dim)

        """
        # src len x batch_size x hidden_size
        encoder_outputs, h_enc = self.encode(src)

        if self.forward_mode == 'greedy':
            max_len = trg.shape[0]
            input_trg = trg[0].unsqueeze(0)
            h_0 = h_enc

            outputs = torch.zeros((max_len, trg.shape[1],
                              self.output_dim)).to(self.device)

            for t in range(1, max_len):
                mask = self.create_mask(src, input_trg)
                decoder_outputs, h_0 = self.decode( input_trg, encoder_outputs,  h_0 = h_0, **mask)

                teacher_force = random.random() < teacher_forcing_ratio
                top1 = decoder_outputs[-1].argmax(-1)
                input_trg = (trg[t] if teacher_force else top1).unsqueeze(0)
                outputs[t-1] = decoder_outputs[-1]

            return outputs

        elif self.forward_mode == 'next_word':
            mask = self.create_mask(src, trg)

            decoder_outputs, h_0 = self.decode( trg, encoder_outputs,  h_0=h_enc, **mask)

            return decoder_outputs

    def encode(self, src, **kwargs):
        src_embeded = self.dropout(self.src_embedding(src))
        encoder_outputs, h = self.encoder(src_embeded)
        # src len x batch_size x hidden_size
        encoder_outputs = self.fc(encoder_outputs)
        return encoder_outputs, self.enc2dec(h.permute(1,2,0)).permute(2,0,1).contiguous()

    def decode(self,trg, encoder_outputs,  mask = None, h_0 = None, **kwargs):
        trg_embeded = self.dropout(self.trg_embedding(trg))
        # trg len x batch_size x hidden_size
        if h_0 != None:
            decoder_outputs, h_0 = self.decoder(trg_embeded, h_0)
        else:
            decoder_outputs, h_0 = self.decoder(trg_embeded)

        scores = torch.bmm(decoder_outputs.permute(1, 0, 2), encoder_outputs.permute(
            1, 2, 0),) # batch_size  x trg len x src len
        scores.masked_fill(mask == 0, float('-inf'))
        weighted = torch.bmm(scores, encoder_outputs.permute(1, 0, 2)).permute(
            1, 0, 2)  # batch_size  x trg len x hidden_size

        decoder_outputs = torch.cat([decoder_outputs, weighted], dim=-1)
        decoder_outputs = self.to_trg_vocab_size(decoder_outputs)

        return decoder_outputs, h_0

