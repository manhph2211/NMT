import spacy
import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
import en_core_web_sm
import de_core_news_sm


spacy_de=de_core_news_sm.load()
spacy_en=en_core_web_sm.load()

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]