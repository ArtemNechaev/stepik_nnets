from turtle import forward
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class OnlyGRU(nn.Module):
    def __init__(self, input_dim, output_dim, emd_size, hidden_size,  pad_idx, device, dropout=0.3) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emd_size = emd_size

        self.device = device

        self.src_embedding = nn.Embedding(
            input_dim, emd_size, padding_idx=pad_idx)
        self.trg_embedding = nn.Embedding(
            output_dim, emd_size, padding_idx=pad_idx)
        self.encoder = nn.GRU(emd_size, hidden_size, 2, bidirectional=True)
        self.decoder = nn.GRU(emd_size, hidden_size, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.to_trg_vocab_size = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, src, trg, *args):
        """
        src - src len x batch_size
        trg - trg sent len x batch_size


        return:
            outputs - trg sent len x batch_size x trg_vocab_size (output_dim)

        """
        # src len x batch_size x hidden_size
        encoder_outputs = self.encode(src)

        decoder_outputs, att = self.decode(encoder_outputs, trg)

        return decoder_outputs

    def encode(self, src):
        src_embeded = self.dropout(self.src_embedding(src))
        encoder_outputs, _ = self.encoder(src_embeded)
        # src len x batch_size x hidden_size
        encoder_outputs = self.fc(encoder_outputs)
        return encoder_outputs

    def decode(self, encoder_outputs, trg):
        trg_embeded = self.dropout(self.trg_embedding(trg))
        # trg len x batch_size x hidden_size
        decoder_outputs, _ = self.decoder(trg_embeded)

        scores = torch.bmm(decoder_outputs.permute(1, 0, 2), encoder_outputs.permute(
            1, 2, 0),)  # batch_size  x trg len x src len

        weighted = torch.bmm(scores, encoder_outputs.permute(1, 0, 2)).permute(
            1, 0, 2)  # batch_size  x trg len x hidden_size

        decoder_outputs = torch.cat([decoder_outputs, weighted], dim=-1)
        decoder_outputs = self.to_trg_vocab_size(decoder_outputs)

        return decoder_outputs, scores

    def predict(self, src, beam_size=5, max_len=20, device=None):

        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        src = src.unsqueeze(1).repeat(1, beam_size).to(device)

        # outputs [sent len, 1, enc hid dim * 2] #hidden = [1, dec hid dim]
        encoder_outputs = self.encode(src)

        last_scores = torch.zeros((beam_size, 1)).to(device)

        # tensor to store decoder outputs
        outputs = torch.zeros(beam_size, max_len).to(device)
        outputs[:,0] = src[0, :]
        # tensor to store attention
        #attentions = torch.zeros(max_len, beam_size, src.shape[0]).to(device)

        for i in range(1, max_len):
            dec_logits,  att = self.decode(encoder_outputs, outputs[:,:i].permute(1,0))
            dec_logits = dec_logits[-1]

            scores = F.log_softmax(dec_logits, 1)
            ind = scores.argsort(1, descending=True)
            scores = torch.gather(scores, 1, ind)[:, :beam_size]
            ind = ind[:, :beam_size]

            scores += last_scores
            order_scores = scores.flatten().argsort(descending=True)
            if i == 0:
                output = ind.flatten()[:beam_size]
            else:
                output = ind.flatten()[order_scores][:beam_size]

            beam_ids = torch.div(
                order_scores[:beam_size], beam_size, rounding_mode='floor')
            hidden = hidden[beam_ids]
            outputs = outputs[beam_ids]
            att = att[beam_ids]

            new_scores = scores.flatten(
            )[order_scores][:beam_size].unsqueeze(1)
            new_scores[torch.where(outputs[:, i-1] == self.eos_idx)] = 0
            lens_hyp = ((outputs != self.eos_idx) & (
                outputs != self.pad_idx)).sum(1, keepdim=True)
            last_scores = (last_scores[beam_ids] +
                           new_scores)/(lens_hyp + 1) ** 0.5

            outputs[:, i+1] = output
            #attentions[i] = att

            if torch.all(output == self.eos_idx):
                return outputs[:, :i+1], last_scores, att

        return outputs, last_scores, att
