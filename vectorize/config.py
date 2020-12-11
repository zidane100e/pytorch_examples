import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

START_TAG = '[CLS]' # '<START>'
STOP_TAG = '[SEP]' #'<STOP>'
PAD_TAG = '[PAD]' 
UNK_TAG = '[UNK]'
