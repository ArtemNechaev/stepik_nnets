from msilib import sequence
from typing import Sequence
import torch

def data_to_device (data, device):
    if isinstance(data, Sequence):
        data = (d.to(device) for d in data)
    else:
        data = data.to(device)
    return data

def train(model, iterator, optimizer, criterion, clip, device = None):
    
    model.train()
    
    epoch_loss = 0

    if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    
    for i, batch in enumerate(iterator):
        
        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg[:-1], 0.4)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        

        output = output.permute(1,2,0)
        trg = trg[1:].permute(1,0)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device=None):
    
    model.eval()
    
    epoch_loss = 0

    if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, trg = batch
            src = src.to(device)
            trg = trg.to(device)


            output = model(src, trg[:-1], 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            #output = output[1:].view(-1, output.shape[-1])
            #trg = trg[1:].view(-1)

            output = output.permute(1,2,0)
            trg = trg[1:].permute(1,0)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def predict_with_model(model, iterator, device = None):
    if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch in iterator:

        src, trg = data_to_device(batch, device)
        