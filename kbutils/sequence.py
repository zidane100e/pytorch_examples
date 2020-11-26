import torch

def int2onehot(x, n_class):
    ret = torch.zeros(n_class)
    ret[x] = 1
    return ret