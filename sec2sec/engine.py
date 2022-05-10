import matplotlib
from IPython.display import clear_output
import matplotlib.pyplot as plt
from typing import Sequence, Dict
from torchtext.vocab import Vocab
import torch

import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})


def data_to_device(data, device):

    if isinstance(data, Sequence):
        data = tuple(d.to(device) for d in data)
    elif isinstance(data, Dict):
        data = {k: v.to(device) for k, v in data.items()}
    else:
        data = data.to(device)
    return data


def train(model, iterator, optimizer, criterion, clip, train_history=None, valid_history=None, teacher_forcing_ratio=0.4):

    model.train()

    epoch_loss = 0
    history = []

    device = model.device

    for i, batch in enumerate(iterator):

        batch = data_to_device(batch, device)

        optimizer.zero_grad()

        if isinstance(batch, Sequence):
            output = model(*batch, teacher_forcing_ratio=teacher_forcing_ratio)
            trg = batch[1]
        elif isinstance(batch, Dict):
            output = model(
                **batch, teacher_forcing_ratio=teacher_forcing_ratio)
            trg = batch['labels']
        else:
            output = model(batch, teacher_forcing_ratio=teacher_forcing_ratio)
            trg = batch[1:]

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[:-1].permute(1, 2, 0)
        trg = trg[1:].permute(1, 0)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        if (train_history is not None) & (valid_history is not None):
            history.append(loss.cpu().data.numpy())
            if (i+1) % 10 == 0:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

                clear_output(True)
                ax[0].plot(history, label='train loss')
                ax[0].set_xlabel('Batch')
                ax[0].set_title('Train loss')

                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
                ax[1].plot(valid_history, label='general valid history')
                plt.legend()

                plt.show()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    device = model.device

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            batch = data_to_device(batch, device)

            if isinstance(batch, Sequence):
                output = model(
                    *batch, teacher_forcing_ratio=0)
                trg = batch[1]
            elif isinstance(batch, Dict):
                output = model(
                    **batch, teacher_forcing_ratio=0)
                trg = batch['labels']
            else:
                output = model(
                    batch, teacher_forcing_ratio=0)
                trg = batch[1:]

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[:-1].permute(1, 2, 0)
            trg = trg[1:].permute(1, 0)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
