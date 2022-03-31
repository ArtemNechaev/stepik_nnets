import torch


def train(model, iterator, optimizer, criterion, clip, device = None):
    
    model.train()
    
    epoch_loss = 0

    if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i, batch in enumerate(iterator):
        
        src, trg = batch
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        output, attetion = model(src, trg, 0.4)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        

        output = output[1:].permute(1,2,0)
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
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src, trg = batch
            src = src.to(device)
            trg = trg.to(device)


            output, attention = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            #output = output[1:].view(-1, output.shape[-1])
            #trg = trg[1:].view(-1)

            output = output[1:].permute(1,2,0)
            trg = trg[1:].permute(1,0)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)