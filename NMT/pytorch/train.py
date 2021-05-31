from model import TransformerModel
from engine import train_fn, val_fn
from torch import optim
import pandas as pd
import config
from dataset import *
import torch.nn as nn
import torch

NMT_model = TransformerModel(INPUT_SIZE, config.EMBED_SIZE, config.HIDDEN_SIZE, OUTPUT_SIZE, config.NUM_LAYER,
         
                            config.MAX_LENGTH, PAD_token, SOS_token, EOS_token).to(config.device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


NMT_model.apply(init_weights)
NMT_model.load_state_dict(torch.load(config.model_save_path))
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
optimizer = optim.Adam(NMT_model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)


best_loss_val = 99999
log=[]
    
for epoch in range(config.N_EPOCHS+1):
  train_loss = train_fn(NMT_model, train_iter,criterion,optimizer)
  val_loss = val_fn(NMT_model, valid_iter,criterion)
  
  log_epoch = {"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss}
  log.append(log_epoch)
  df = pd.DataFrame(log)
  df.to_csv(config.save_log) 
  print("Epoch {} || epoch_train_loss: {:.4f} || Epoch_val_loss: {:.4f} ".format(epoch + 1,train_loss, val_loss))
  
  if val_loss < best_loss_val:
    best_loss_val = val_loss
    torch.save(NMT_model.state_dict(),config.model_save_path)
    scheduler.step(val_loss)


