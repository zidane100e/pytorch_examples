#!/usr/bin/env python
# coding: utf-8

# ## Dataloader
# * reference : Natural language processing with PYTORCH
# 1. load data file into a dataframe
# 2. modify text with right form using tokenizer or corpus
# 3. change modified text into vectors

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


# In[9]:


import os, re, string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import notebook


# In[6]:


if torch.cuda.device_count()>1:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif torch.cuda.device_count()>0:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[162]:


import spacy
from torchtext import data as ttdata
from torchtext.data import Dataset, Example, Field
from torchtext.data import Iterator, BucketIterator

class TextData():    
    def __init__(self, df_train, df_test, df_valid=None):
        self.df = {}
        self.df['train'] = df_train
        self.df['test'] = df_test
        if df_valid is None:
            self.df['valid'] = df_test
        else:
            self.df['valid'] = df_valid
                    
        self.init_field()
        self.data_setting = 'train'
        self.dataset = {}
    
    def set_data_setting(self, setting):
        self.data_setting = setting
        
    def init_field(self):
        self.field_x_name = 'data_x'
        self.field_x = Field(sequential=True, use_vocab=True,
                 tokenize='spacy', lower=True,
                 batch_first=True, fix_length=100,
                 init_token='<SOS>', eos_token='<EOS>'
                 )
        self.field_y_name = 'data_y'
        self.field_y = Field(sequential=False, use_vocab=False,
                  batch_first=False, is_target=True)
        self.fields = ((self.field_x_name, self.field_x),
                     (self.field_y_name, self.field_y))
    
    def _get_dataset(self, setting='train'):
        self.data_setting = setting
        train_data = []
        for ii in range(len(self.df[setting])):
            temp = Example.fromlist(
                        self.df[setting].iloc[ii].values.tolist(),
                              self.fields)
            train_data.append(temp)
        self.dataset[setting] = Dataset(train_data, fields=self.fields)
        return self.dataset[setting]
        
    def _generate_vocab(self, min_freq=10, max_size=20000):
        try:
            self.dataset
        except:
            self._get_dataset('train')
        self.field_x.build_vocab(self.dataset['train'], 
                                 min_freq=min_freq, 
                                 max_size=max_size)
        self.vocab = self.field_x.vocab
        return self.vocab
    
    def get_batch(self, batch_size, setting='train', 
                  min_freq=10, max_size=20000):
        try: 
            self.dataset[setting]
        except:
            self._get_dataset(setting)
            self._generate_vocab(min_freq, max_size)
        try:
            self.iter
        except:
            self.iter = Iterator(self.dataset[setting], 
                                 batch_size=batch_size)
        for batch in iter(self.iter):
            xs = getattr(batch, 'data_x')
            ys = getattr(batch, 'data_y')
            xs = xs.to(device)
            ys = ys.to(device)
            yield xs, ys


if __name__ == 'main':
    # ## Preprocess

    # In[160]:


    """
    data shape
    rating, review, split
    negative, "sentence", train
    positive, "sentence", train
    """
    f1_s = "/home/bwlee/data/yelp_review_polarity_csv/reviews_with_splits_full.csv"

    #df = pd.read_csv(f1_s, header=0)
    #df = pd.read_csv(f1_s, header=0, nrows=100)
    df = pd.read_csv(f1_s, header=0, nrows = 100, 
                     skiprows=lambda x: x%7000>0)
    label0 = []
    for i in df.index: # change this for getting info based on previous day market move
        if df['rating'].iloc[i] == 'positive':
            label0.append(1)
        else:
            label0.append(0)
    df['rating'] = label0

    train_df = df[df['split']=='train'][['review','rating']] 
    val_df = df[df['split']=='val'][['review','rating']]
    test_df = df[df['split']=='test'][['review','rating']]
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)


    # In[165]:


    data = TextData(train_df, test_df)
    batches = data.get_batch(10)
    for x, y in batches:
        print(y)


    # ## dataset test

    # ## Dataset and DataLoader
    # * Dataset gets data
    # * map-style, iterable style
    #     * map-style(Dataset) :  needs __getitem__(), __len__()
    #     * iterable style(IterableDataset) : needs __iter__()
    # * DataLoader transform Dataset into batch and tensors
    # ```python
    # torch.utils.data.Dataset
    # torch.utils.data.TensorDataset(*tensors)
    # # *tensors (Tensor) – tensors that have the same size of the first dimension.
    # ```
    # * TensorDataset can be used for multiple data with a condition for dimension.

    # ### DataLoader
    # * if it None, batch is not applied
    # * defau;t batch_size is 1
    # * batch_sampler works for map-style datasets

    # ## torchtext : DataLoader, Iterator
    # * There's another dataloader for text in torchtext
    # * This has various open dataset
    # * It is still cumbersome in reading from dataframe
    # * In case, if you read directly from TSV, CSV, JSON, ...  
    # You can use TabularDataset
    # * torchtext.data.Dataset requires Example type of data
    #     * You need to make Example for each data instance  
    #       and give it as input    

    # we can use several type of tokenizer.
    # In here, we use spacy

    # If dataset is made, we can generate vocabulary.  
    # If a vocabulary is established, dataloader vectorize  
    # each sentence.

    # In[166]:


    data.vocab.itos[58]


    # In[167]:


    data.vocab.stoi['i']


    # In[ ]:




