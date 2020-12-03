"""
some functions for pre-trained model usages
"""

#put import sentences
import torch

def get_attention_mask(token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()