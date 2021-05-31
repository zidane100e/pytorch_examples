import torch

def int2onehot(x, n_class):
    """
    return integet input to one hot encoding
    :param x: integer to set 1
    :param n_class: number of classes
    """
    ret = torch.zeros(n_class)
    ret[x] = 1
    return ret