import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, embedding=None):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        if embedding != None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # mask = [batch size, src sent len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src sent len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class MyAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self,  hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # mask = [batch size, src sent len]
        encoder_outputs = self.fc(encoder_outputs)
        encoder_outputs = encoder_outputs / torch.linalg.norm(encoder_outputs, dim=-1, keepdim=True)
        hidden = hidden/torch.linalg.norm(hidden, dim=-1, keepdim=True)

        attention = torch.bmm(hidden.unsqueeze(
            1), encoder_outputs.permute(1, 2, 0)).squeeze(1)

        # attention = [batch size, src sent len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, embedding=None):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        if embedding != None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(output_dim, emb_dim)

        self.combine = nn.Linear(enc_hid_dim * 2 + emb_dim, enc_hid_dim * 2)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear((enc_hid_dim * 2) +
                             dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_input, encoder_outputs, mask):

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_input = [src sent len, batch size]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        # mask = [batch size, src sent len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src sent len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src sent len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        #assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src sent len, batch size]
        # src_len = [batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg = torch.zeros((100, src.shape[1])).long().fill_(
                self.sos_idx).to(src.device)
        else:
            inference = False

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(self.device)

        # tensor to store attention
        #attentions = torch.zeros(
            #max_len, batch_size, src.shape[0]).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        output = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src sent len]

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(
                output, hidden, src, encoder_outputs, mask)
            outputs[t] = output
            #attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            output = (trg[t] if teacher_force else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t]

        return outputs

    def predict(self, src, beam_size=5, max_len=20, device=None):

        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        src = torch.LongTensor(src).unsqueeze(
            1).repeat(1, beam_size).to(device)
        # outputs [sent len, 1, enc hid dim * 2] #hidden = [1, dec hid dim]
        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)  # beam_size x src_len

        output = src[0, :]  # [beam_size]

        last_scores = torch.zeros((beam_size, 1)).to(device)
        # tensor to store decoder outputs
        outputs = torch.zeros(beam_size, max_len, dtype=torch.long).to(device)

        # tensor to store attention
        attentions = torch.zeros(max_len, beam_size, src.shape[0]).to(device)

        for i in range(max_len):
            dec_logits, hidden, att = self.decoder(
                output, hidden, src, encoder_outputs, mask)

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

            outputs[:, i] = output
            attentions[i] = att

            if torch.all(output == self.eos_idx):
                return outputs[:, :i], last_scores, attentions.permute(1, 0, 2)[:, :i]

        return outputs, last_scores, attentions.permute(1, 0, 2)[:, :i]
