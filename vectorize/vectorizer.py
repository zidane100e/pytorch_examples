"""
vectorizer changes tokens to ids and get embeds from pre-trained model
This can be used for both of tokens and labels
"""

from collections import OrderedDict

import pandas as pd
import numpy as np
import copy as cp

from config import *
from tokenizer import KBTokenizer

def load_glove(f1_s, n_tag = -1):
    """
    n_tag :  limit vocabulary
    """
    special_chars = [UNK_TAG, PAD_TAG, START_TAG, STOP_TAG]
    tag2ix, ix2tag, tag2vec = {}, {}, {}
    for ix, char in enumerate(special_chars):
        tag2ix[char] = ix
        ix2tag[ix] = char
        tag2vec[char] = None
    ix = len(special_chars)
    with open(f1_s) as f1:
        for line in f1:
            if n_tag >= 0 and ix >= n_tag:
                break                    
            elms = [ x.strip().lower() if i==0 else float(x.strip()) for i, x in enumerate(line.split()) ]
            tag = elms[0]
            val = elms[1:]
            tag2vec[tag] = val
            tag2ix[tag] = ix
            ix2tag[ix] = tag
            ix += 1
                
    n_embed = len(tag2vec['the'])
    tag2vec[PAD_TAG] = np.zeros(n_embed)
    tag2vec[UNK_TAG] = np.zeros(n_embed)
    #tag2vec[UNK_TAG] = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), n_embed)
    tag2vec[START_TAG] = np.zeros(n_embed)
    tag2vec[STOP_TAG] = np.zeros(n_embed)
    return tag2vec, tag2ix, ix2tag

class Vectorizer():
    """
    transform words into vectors
    """
    def __init__(self, tagmap, min_tag_occur=-1,
                 lower_tag=False, flag_special=True):
        """
        :param tagmap: it is assumed that tagmap is arranged by count
        """
        have_special_chars = True
        self.tags = list(tagmap.keys())
        self.pre_trained = None
        if flag_special is True:
            for char in [UNK_TAG, PAD_TAG, START_TAG, STOP_TAG]:
                check = (char in self.tags)
                have_special_chars &= check
                if have_special_chars is False:
                    raise Exception("Add special character %s"%char)
        
        self.tag2ix, self.ix2tag, self.tag2vec = {}, {}, {}
        self.lower_tag = lower_tag

        self.n_tag = 0
        for tag, count in tagmap.items():
            if count < min_tag_occur:
                continue
            self.tag2ix[tag] = self.n_tag
            self.ix2tag[self.n_tag] = tag
            self.n_tag += 1
    
    def add_dict(self, new_dict, lower_tag=True):
        """
        usages: union of training data and additional dictionary
        word_vectorizer = Vectorizer(tokensmap, min_tag_occur=3, lower_tag=True)
        word_vectorizer.add_dict(glove2vec.keys())
        """
        set1 = set(self.tag2ix)
        set2 = set(new_dict)
        setc = set2 - set1
        
        for x in setc:
            self.tag2ix[x] = self.n_tag
            self.ix2tag[self.n_tag] = x
            self.n_tag += 1
        self.tags = list(self.tag2ix.keys())
    
    @classmethod
    def from_map(cls, tag2ix, lower_tag=True):
        """
        usages:
        with open(f_s, 'rb') as f1:
        tag2ix = pk.load(f1)
        word_vectorizer = Vectorizer.from_map(tag2ix)
        """
        # for dummy instance
        tagmap = {UNK_TAG: 1e6, PAD_TAG: 1e6, START_TAG: 1e6, STOP_TAG: 1e6} 
        inst1 = cls(tagmap, -1, lower_tag)
        inst1.tag2ix = cp.copy(tag2ix)
        inst1.ix2tag = {ix: word for word, ix in tag2ix.items()}
        inst1.n_tag = len(tag2ix)
        inst1.tags = list(tag2ix.keys())
        return inst1
    
    @classmethod
    def from_glove(cls, f1_s, n_tag = -1):
        """
        usages:
        word_vectorizer, glove2vec = Vectorizer.from_glove(f_s)
        """
        tag2vec, tag2ix, ix2tag = load_glove(f1_s, n_tag)
        inst1 = Vectorizer.from_map(tag2ix, True)
        inst1.tag2vec = tag2vec
        inst1.pre_trained = 'glove'
        inst1.n_tag = len(tag2vec)
        inst1.tags = list(tag2ix.keys())
        return inst1

    @classmethod
    def from_bert(cls, pre_trained, len_sent = 512, lower_tag=False):
        tagmap = {UNK_TAG: 1e6, PAD_TAG: 1e6, START_TAG: 1e6, STOP_TAG: 1e6} 
        inst1 = cls(tagmap, -1, lower_tag)
        inst1.pre_trained = pre_trained
        inst1.len_sent = len_sent
        inst1.tokenizer = KBTokenizer(pre_trained).tokenizer
        if pre_trained == 'sktkobert':
            inst1.n_tag = len(inst1.tokenizer.vocab)
            inst1.ix2tag = inst1.tokenizer.vocab.idx_to_token
        elif pre_trained in ['bert-multi', 'kbalbert']:
            inst1.n_tag = inst1.tokenizer.vocab_size
            inst1.ix2tag = inst1.tokenizer.ids_to_tokens
        return inst1

    def _get_attention_mask(self, len_sent, valid_length):
        attention_mask = np.zeros(len_sent)
        attention_mask[:valid_length] = 1
        return attention_mask
    
    def get_ids_bert(self, seq):
        """
        Padding, sentence length is determined in class initialization
        """
        if type(seq) is str:
            seq = self.tokenizer.tokenize(seq)

        if self.pre_trained == 'bert-multi':
            ix_PAD = 0
            ix_UNK = 100
            ix_CLS = 101
            ix_SEP = 102
        elif self.pre_trained == 'sktkobert':
            ix_UNK = 0
            ix_PAD = 1
            ix_CLS = 2
            ix_SEP = 3
        elif self.pre_trained == 'kbalbert':
            ix_PAD = 0
            ix_UNK = 1
            ix_CLS = 2
            ix_SEP = 3
        else:
            raise Exception('Check bert name')
        input_ids = [ self.tokenizer.vocab[token] if token in self.tokenizer.vocab else ix_UNK for token in seq ]
        
        if len(input_ids) < self.len_sent-2: # [CLS], [SEP] need to be attached 
            length = len(input_ids) + 2
            input_ids = [ix_CLS] + input_ids + [ix_SEP] + [ix_PAD]*(self.len_sent-2-len(input_ids))
        else:
            length = self.len_sent
            input_ids = [ix_CLS] + input_ids[:self.len_sent-2] + [ix_SEP]
        attention_mask = self._get_attention_mask(len(input_ids), length)
        type_id = [0] * self.len_sent
        return (input_ids, attention_mask, type_id)

    def get_ids_non_bert(self, seq, add_pad=False, add_beginend=False, len_sent=-1):
        """
        :param seq: sequence of tokens
        :return: gives id of seq
        """
        ret = []
        for x in seq:
            if self.lower_tag:
                x = x.lower()
            if x in self.tags:
                ret.append(self.tag2ix[x])
            else:
                ret.append(self.tag2ix[UNK_TAG])
                
        if add_beginend:
            if len_sent>0 and len(ret) > len_sent-2:
                ret = ret[:len_sent-2]
            ret = [self.tag2ix[START_TAG]] + ret + [self.tag2ix[STOP_TAG]]
        else:
            if len_sent>0 and len(ret) > len_sent:
                ret = ret[:len_sent]

        if len_sent>0 and add_pad:
            ret += [self.tag2ix[PAD_TAG]]*(len_sent-len(ret))

        return ret
        
    def get_ids(self, seq, add_pad=False, add_beginend=False,
                len_sent=-1):
        return self(seq, add_pad, add_beginend, len_sent)
    
    def __call__(self, seq, add_pad=False, add_beginend=False, len_sent=-1):
        if self.pre_trained is not None and 'bert' in self.pre_trained.lower():
            return self.get_ids_bert(seq)
        else:
            return self.get_ids_non_bert(seq, add_pad, add_beginend, len_sent)
    
    def to_str(self, vecs):
        return [ self.ix2tag[x] for x in vecs ]
    
if __name__ == '__main__':
    from tokenizer import KBTokenizer
    tagmap = {'a': 10, 'b': 12, 'c': 15}
    vectorizer = Vectorizer(tagmap, flag_special=False)
    print(vectorizer.tag2ix)
    seq = ['a', 'b', 'c', 'c', 'a']
    ids = vectorizer(seq)
    print(ids)

    tokenizer = KBTokenizer('spacy')
    str1 = 'I love you.'
    seq1 = tokenizer.tokenize(str1)
    print('~~~~', seq1, type(seq1))
    #seq = ['i', 'miss', 'you']
    glove_s = '/home/bwlee/data/glove/glove.6B.100d.txt'
    glove_vectorizer = Vectorizer.from_glove(glove_s)
    ids = glove_vectorizer(seq1)
    print(ids)

    pre_trained = 'bert-multi'
    tokenizer = KBTokenizer(pre_trained)
    vectorizer = Vectorizer.from_bert(pre_trained)
    str1 = '난 당신을 사랑합니다.'
    seq1 = tokenizer.tokenize(str1)
    print('seq', seq1)
    ids = vectorizer(seq1)
    print(ids)

    pre_trained = 'sktkobert'
    tokenizer = KBTokenizer(pre_trained)
    vectorizer = Vectorizer.from_bert(pre_trained)
    str1 = '난 당신을 사랑합니다.'
    seq1 = tokenizer.tokenize(str1)
    print('seq', seq1)
    ids = vectorizer(seq1)
    print(ids)
    
    pre_trained = 'kbalbert'
    tokenizer = KBTokenizer(pre_trained)
    vectorizer = Vectorizer.from_bert(pre_trained)
    str1 = '난 당신을 사랑합니다.'
    seq1 = tokenizer.tokenize(str1)
    print('seq', seq1)
    ids = vectorizer(seq1) 
    print(ids)
    print('back', vectorizer.to_str(ids[0]))

    vectorizer = Vectorizer.from_bert(pre_trained)
    ids = vectorizer(seq1, add_pad=True) 
    print('pad', ids)
    print('pad back', vectorizer.to_str(ids[0]))

    vectorizer = Vectorizer.from_bert(pre_trained)
    ids = vectorizer(seq1, add_beginend=True) 
    print('begin', ids)
    print('begin back', vectorizer.to_str(ids[0]))

    vectorizer = Vectorizer.from_bert(pre_trained)
    ids = vectorizer(seq1, add_pad=True, add_beginend=True)
    print('pad begin', ids)
    print('pad begin back', vectorizer.to_str(ids[0]))  
    
    exit()

    ret1 = vectorizer(seq1)
    print(ret1)
    print(vectorizer.to_str(ret1))
    #ret2 = char_vectorizer(seq2)
    #print(ret2)
    print(glove2vec['the'])
    print(vectorizer.tag2vec['the'])
    exit()
   
    print(char_vectorizer.to_str(ret2))
    #print([ char_vectorizer.ix2tag[x] for x in ret2 ])
    ret1 = word_vectorizer(seq1, add_beginend=True)
    ret2 = word_vectorizer(seq1, add_pad=True, len_sent=10, add_beginend=True)
    ret3 = word_vectorizer(seq1, add_pad=True, len_sent=10)
    print(word_vectorizer.to_str(ret1))
    print(word_vectorizer.to_str(ret2))
    print(word_vectorizer.to_str(ret3))
