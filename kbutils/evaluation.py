import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def argmax(vec):
    """
    :return: the argmax as a python int
    """
    _, idx = torch.max(vec, 1)
    return idx.item()

def compute_accuracy(y_pred, y_target):
    """
    f1_score = precision = recall in case len(pred) == len(target)
    :return: (accuracy, precision)
    """
    y_pred_indices = (torch.sigmoid(y_pred)>0.5).cpu().long()#.max(dim=1)[1]
    
    # calculate accuracy
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    accuracy = n_correct / len(y_pred_indices) * 100
    
    # calculate precision
    pred_true = torch.eq(y_pred_indices, 1)
    tgt_true = torch.eq(y_target, 1)
    pred_tgt_true = torch.eq(pred_true, tgt_true).sum().item()
    precision = pred_tgt_true/len(y_target) * 100  
    
    return accuracy, precision

def accuracy(out, tgt):
    """
    :param out: softmax output
    """
    preds = np.argmax(out, axis=-1)
    return accuracy_score(tgt, preds)

if __name__ == '__main__':
    int2onehot(3, 10)
