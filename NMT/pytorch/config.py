train_en = '../data/train/train.en'
val_en = '../data/val/validation.en'
test_en = '../data/test/test.en'


train_vi = '../data/train/train.vi'
val_vi = '../data/val/validation.vi'
test_vi = '../data/test/test.vi'


device = 'cuda'


LR = 0.0003
N_EPOCHS = 100
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYER = 6
MAX_LENGTH = 64


model_save_path = '../weights/nmt.pth'
save_log = '../weights/log.csv'

