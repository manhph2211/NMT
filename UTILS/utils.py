import torch
from torchtext.data import Field, BucketIterator
import torchtext
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


def split_data(BATCH_SIZE,device,spacy_de,spacy_en):

	SRC = Field(tokenize = tokenize_de, 
	            init_token = '<sos>', 
	            eos_token = '<eos>', 
	            lower = True, 
	            batch_first = True)

	TRG = Field(tokenize = tokenize_en, 
	            init_token = '<sos>', 
	            eos_token = '<eos>', 
	            lower = True, 
	            batch_first = True)

	# load the Multi30k dataset
	train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))
	  
	# load the Multi30k dataset
	SRC.build_vocab(train_data, min_freq = 2)
	TRG.build_vocab(train_data, min_freq = 2)

	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
	    (train_data, valid_data, test_data), 
	     batch_size = BATCH_SIZE,
	     device = device)

	return train_iterator, valid_iterator, test_iterator, SRC, TRG