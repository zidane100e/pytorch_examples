"""
text --> tokenizer --> vectorizer --> embedder : context
(BERT embedder include vectorizer 
"""
from config import *
from vectorizer import *
from tokenizer import KBTokenizer

class Embedder():
    def __init__(self, vectorizer=None, tokenizer=None,
                 dim_embed=200):
        """
        :param tokenizer: KB 
        """
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer 
        self.pre_trained = pre_trained = vectorizer.pre_trained
        self.n_tag = self.vectorizer.n_tag
        
        if 'bert' in pre_trained.lower():
            self.tag2vec = None
            import sys
            if pre_trained == 'bert-multi':
                from transformers import BertModel, BertConfig
                bert_config = BertConfig.from_pretrained('bert-base-multilingual-cased',
                                                    output_hidden_states=True)
                self.bert = BertModel(bert_config).to(device)
            elif pre_trained == 'sktkobert':
                from kobert.pytorch_kobert import get_pytorch_kobert_model
                #sys.path.append('/home/bwlee/work/codes/sentence_similarity/kobert')
                #from pytorch_kobert3 import get_pytorch_kobert_model
                self.bert, _ = get_pytorch_kobert_model()
                self.bert = self.bert.to(device)
            if pre_trained == 'kbalbert':
                sys.path.append('/home/bwlee/work/codes/KB-ALBERT-KO/kb-albert-char/')
                from transformers import AlbertModel
                kbalbert_path = '/home/bwlee/work/codes/KB-ALBERT-KO/kb-albert-char/model'
                self.bert = AlbertModel.from_pretrained(kbalbert_path, 
                                    output_hidden_states=True)
                self.bert = self.bert.to(device)
        else:
            self.tag2vec = self.vectorizer.tag2vec
            self.n_vocab = len(self.vectorizer.tag2vec)
            if pre_trained == '':
                self.embed = nn.Embedding(num_embeddings=self.n_tag,
                                          embedding_dim=dim_embed,
                                          padding_idx=self.tag2ix[PAD_TAG])
    def set_embed(self, weights=None, bias=None):
        if weights is not None:
            self.embed.weight.data = weights
        if bias is not None:
            self.embed.bias.data = bias

    def __call__(self, text_arr):
        """
        check type_ids=None gives different result in bert-multi
        :param text_arr: accepts text in iterable form like batch
        """
        if type(text_arr) is str:
            print('warning: text should be in batch form')
            text_arr = [text_arr]

        if self.pre_trained == '':
            return self._call_manual(text_arr)
        elif self.pre_trained == 'glove':
            return self._call_glove(text_arr)
        elif 'bert' in self.pre_trained:
            return self._call_bert(text_arr)

    def _call_manual(self, text_arr):
        """
        :param text_arr: accepts text in iterable form like batch
        """
        idss = []
        for text in text_arr:
            seq = self.tokenizer.tokenize(text)
            ids = self.vectorizer.get_ids(seq)
            idss.append(ids)
        idss = torch.LongTensor(idss) 
        return self.embed(idss)

    def _call_glove(self, text_arr):
        """
        :param text_arr: accepts text in iterable form like batch
        """
        vecs = []
        dim_glove = len(self.vectorizer.tag2vec['the'])
        zero = [0]*dim_glove
        for text in text_arr:
            seq = self.tokenizer.tokenize(text)
            vec = [ self.vectorizer.tag2vec[token] if token in self.vectorizer.tags else zero for token in seq ]
        vecs.append(vec)
        return torch.tensor(vecs)

    def _call_bert(self, text_arr):
        idss, masks, type_ids = [], [], []
        for text in text_arr:
            seq = self.tokenizer.tokenize(text)
            ids, mask, type_id = self.vectorizer.get_ids_bert(seq)
            idss.append(ids)
            masks.append(mask)
            type_ids.append(type_id)
        idss = torch.tensor(idss).to(device)
        masks = torch.tensor(masks).to(device)
        type_ids = torch.tensor(type_ids).to(device)
        #type_ids = None # bert-multi gives different values
        _, _, hiddens = self.bert(idss, attention_mask=masks, token_type_ids=type_ids) #kbalbert
        context = torch.mean(hiddens[-2], dim=1)
        return context

if __name__ == '__main__':
    from vectorizer import Vectorizer

    texts = ['i love you', 'what do you want', 'so missed you']
    
    tokenizer = KBTokenizer('spacy')
    glove_s = '/home/bwlee/data/glove/glove.6B.100d.txt'
    vectorizer = Vectorizer.from_glove(glove_s)
    embedder = Embedder(vectorizer, tokenizer)
    vec1 = embedder(texts)
    print(vec1)

    texts = ['브람스를 좋아하세요?', '어서와! 한국은 처음이니?', '대한외국인 퀴즈쇼']

    pre_trained = 'kbalbert'
    tokenizer = KBTokenizer(pre_trained)
    vectorizer = Vectorizer.from_bert(pre_trained, len_sent=100)
    embedder = Embedder(vectorizer, tokenizer)
    vec1 = embedder(texts)
    print(vec1)

    pre_trained = 'sktkobert'
    tokenizer = KBTokenizer(pre_trained)
    vectorizer = Vectorizer.from_bert(pre_trained, len_sent=100)
    embedder = Embedder(vectorizer, tokenizer)
    vec1 = embedder(texts)
    print(vec1)

    pre_trained = 'bert-multi'
    tokenizer = KBTokenizer(pre_trained)
    vectorizer = Vectorizer.from_bert(pre_trained, len_sent=100)
    embedder = Embedder(vectorizer, tokenizer)
    vec1 = embedder(texts)
    print(vec1)

    
    

    
