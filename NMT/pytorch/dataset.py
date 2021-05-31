from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
import config


def filter_len(example):
    return len(example.src) <= config.MAX_LENGTH and len(example.trg) <= config.MAX_LENGTH


SRC = Field(init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(init_token='<sos>', eos_token='<eos>', lower=True)


print('Reading datasets ...')
train_data = TranslationDataset(path=r'../data/train/train', exts=('.en', '.vi'),
                               filter_pred=filter_len, fields=(SRC, TRG))
valid_data = TranslationDataset(r'../data/val/validation', exts=('.en', '.vi'),
                                filter_pred=filter_len, fields=(SRC, TRG))
test_data = TranslationDataset(r'../data/test/test', exts=('.en', '.vi'),
                                filter_pred=filter_len, fields=(SRC, TRG))
print('Datasets reading complete!')


print('Building vocabulary ...')
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print('Vocabulary building complete!')

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=config.BATCH_SIZE, device=config.device
)

INPUT_SIZE = len(SRC.vocab)
OUTPUT_SIZE = len(TRG.vocab)
SOS_token = TRG.vocab.stoi['<sos>']
EOS_token = TRG.vocab.stoi['<eos>']
PAD_token = TRG.vocab.stoi['<pad>']
