from model import TransformerModel
from dataset import *
import torch
from torchtext.data.metrics import bleu_score
import random

def translate(sent,NMTmodel):
    NMTmodel.eval()
    with torch.no_grad():
        src = [SOS_token] + [SRC.vocab.stoi[w] for w in sent] + [EOS_token]
        src = torch.tensor(src, dtype=torch.long, device='cpu')

        wordidx = NMTmodel.inference(src, config.MAX_LENGTH)
        words = []
        for idx in wordidx:
            words.append(TRG.vocab.itos[idx])

    return words


def evaluateRandomly(NMTmodel,data: TranslationDataset, n=5):
    for i in range(n):
        example = random.choice(data.examples)
        src, trg = example.src, example.trg
        print('>', ' '.join(src))
        print('=', ' '.join(trg))
        output_sentence = translate(src,NMTmodel)
        print('<', ' '.join(output_sentence))
        print()


def cal_bleu_score(data,NMTmodel):
    trgs, preds = [], []
    num_example = len(data.examples)
    for example in data.examples:
      src, trg = example.src, example.trg
      pred = translate(src,NMTmodel)[:-1]
      trgs.append([trg])
      preds.append(pred)
    return bleu_score(preds, trgs) 


def evaluate():
    path = config.model_save_path
    print(f'Loading models from {path} ...')
    NMTmodel = TransformerModel(INPUT_SIZE, config.EMBED_SIZE, config.HIDDEN_SIZE, OUTPUT_SIZE, config.NUM_LAYER,
            
                                config.MAX_LENGTH, PAD_token, SOS_token, EOS_token)

    NMTmodel.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    NMTmodel.eval()
    print('Loading models complete!\n')
    print('Calulating BLEU score on valid set ...')
    #valid_bleu = cal_bleu_score(valid_data,NMTmodel)
    
    print('Calulating BLEU score on test set ...')
    #test_bleu = cal_bleu_score(test_data,NMTmodel)

    #print(f'BLEU: {valid_bleu:.2f}')
    #print(f'BLEU: {test_bleu:.2f}')
    print('Evaluating on test set...\n')
    evaluateRandomly(NMTmodel,train_data)

    print('Please enter sentence to be translated:\n')
    while True:
        s = input()
        if s == "q":
          break
        output_sent = translate(s,NMTmodel)
        print('<', ' '.join(output_sent), '\n')


if __name__ == '__main__':
  evaluate()
