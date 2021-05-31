import os
import numpy as np 
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torchtext.datasets import TranslationDataset
import config

def load_data(src_file, trg_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        list_src = f.readlines()
    
    with open(trg_file, 'r', encoding='utf-8') as f:
        list_trg = f.readlines()
    
    data  =	[list_src,list_trg]

    return data

if __name__ == '__main__':
  en , vi = load_data(config.test_en, config.test_vi)
  for i,(en_sentence,vi_sentence) in enumerate(zip(en,vi)):
    print('----------------------------------------------')
    print("English: ", en_sentence)
    print("Vietnamese: ", vi_sentence)
    print('----------------------------------------------')
    if i==2:
      break