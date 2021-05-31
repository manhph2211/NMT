import config
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from dataset import INPUT_SIZE,OUTPUT_SIZE


def train_fn(NMT_model, data_loader,criterion,optimizer):
    NMT_model.train()
    device = config.device
    epoch_loss = 0
    for batch in tqdm(data_loader):
        src = batch.src.to(device) 
        trg = batch.trg.to(device) 
        optimizer.zero_grad()
        output = NMT_model(src, trg[:-1])
        # ignore the first sos_token
        output = output.view(-1, output.size(-1))
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        #clip_grad_norm_(NMT_model.parameters(),max_norm=1)
        epoch_loss+=loss.item()
        print(loss.item())
        optimizer.step()
    return epoch_loss/len(data_loader)


def val_fn(NMT_model, data_loader,criterion):
    NMT_model.eval()
    device = config.device
    epoch_loss = 0
    for batch in tqdm(data_loader):
        src = batch.src.to(device) 
        trg = batch.trg.to(device) 
        output = NMT_model(src, trg[:-1])
        # ignore the first sos_token
        output = output.view(-1, output.size(-1))
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        #clip_grad_norm_(NMT_model.parameters(),max_norm=1)
        epoch_loss+=loss.item()
        print(loss.item())
    return epoch_loss/len(data_loader)
