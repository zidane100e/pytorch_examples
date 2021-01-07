import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.attrs import LOWER
from spacy.attrs import ORTH, NORM

try: PAD_TAG
except:
    from config import *

class KBTokenizer():
    def __init__(self, tokenizer_s='spacy'):
        """
        bert-multi, kbalbert : [PAD], [CLS], ...
        :param tokenizer: string to represent tokenizer like 'spacy', 'bert', ...
        Example::
        
        nlp = English()
        tokenizer = nlp.Defaults.create_tokenizer(nlp)      
        tokenizer = Tokenizer(tokenizer)
        """
        if type(tokenizer_s) is str:
            self.tokenizer_s = tokenizer_s
        if tokenizer_s == 'spacy':
            self.nlp = spacy.load("en_core_web_md") # md, large have embed vectors
            self.tokenizer = self.nlp.Defaults.create_tokenizer(self.nlp)
        elif tokenizer_s == 'bert-multi':
            from transformers import BertTokenizer, BertModel, BertConfig
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.vocab = self.tokenizer.vocab
        elif tokenizer_s == 'sktkobert':
            import gluonnlp as nlp
            from kobert.utils import get_tokenizer
            from kobert.pytorch_kobert import get_pytorch_kobert_model
            kobert, vocab = get_pytorch_kobert_model()
            self.tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)
            self.vocab = vocab
        elif tokenizer_s == 'kbalbert':
            import sys
            sys.path.append('/home/bwlee/work/codes/KB-ALBERT-KO/kb-albert-char/')
            from transformers import AlbertModel, TFAlbertModel
            from tokenization_kbalbert import KbAlbertCharTokenizer
            model_path = '/home/bwlee/work/codes/KB-ALBERT-KO/kb-albert-char/model'
            self.tokenizer = KbAlbertCharTokenizer.from_pretrained(model_path)
            self.vocab = self.tokenizer.vocab
        else:
            if type(tokenizer_s) is str:
                from transformers import BertTokenizer, BertModel, BertConfig
                self.tokenizer = BertTokenizer.from_pretrained(tokenizer_s)
                self.vocab = self.tokenizer.vocab
            elif type(tokenizer_s) is not str:
                self.tokenizer = tokenizer_s
                self.tokenizer_s = 'custom'
            else:
                raise Exception('check tokenizer is correctly defined')
        self.pre_trained = self.tokenizer_s

    @classmethod
    def with_tokenizer(cls, tokenizer):
        inst = cls(tokenizer)
        return inst
    
    def tokenize(self, sent, flag_char=False,
                 special_chars=[PAD_TAG, UNK_TAG]):
        """
        tokenize sentence to token
        """
        if self.tokenizer_s == 'spacy':
            for char in special_chars:
                self.nlp.vocab[char]
                
            tokens = [x.text for x in self.tokenizer(sent)]
            if flag_char is False:
                return tokens
            elif flag_char is True:
                chars = [ self.tok2chars(token, special_chars) for token in tokens ]
                return tokens, chars
        elif self.tokenizer_s == 'sktkobert':
            return self.tokenizer(sent)
        #elif self.tokenizer_s in ['bert-multi', 'kbalbert']:
        elif 'bert' in self.tokenizer_s:
            return self.tokenizer.tokenize(sent)
        
            
    def _index_repetitive(self, obj, element):        
        """
        this finds all occurrences of element in obj
        element is a substring of obj
        """
        indices = []
        ix = 0
        while True:
            try:
                ix = obj.index(element, ix)
                indices.append(ix)
                ix += 1
            except ValueError as e:
                break
        return indices
    
    def tok2chars(self, token, special_chars=[PAD_TAG]):
        """
        * tokenize tokens to chars
        Usually this is unnecessary, because tokenizer do not care about unknown words.
        Vectorizer handles this.
        But some text have unknown words, in such case this can be used.
        """
        chars = []
        indices = [] # [(pos, char)]
        for sp_char in special_chars:
            ixs = self._index_repetitive(token, sp_char)
            for ix in ixs:
                indices.append((ix, sp_char))
        indices = sorted(indices, key= lambda x: x[0])
        
        ii_token, ix_indices = 0, 0
        if len(indices) > 0:
            while ii_token < len(token):
                if ix_indices < len(indices):
                    sp_ix, sp_char = indices[ix_indices]
                else :
                    sp_ix = 1000 # move to else case
                if ii_token == sp_ix:
                    chars.append(sp_char)
                    ii_token += len(sp_char)
                    ix_indices += 1
                elif ii_token > sp_ix:
                    ix_indices += 1
                else:
                    chars.append(token[ii_token])
                    ii_token += 1
            return chars
        else:
            return [ x for x in token ]
        
if __name__ == '__main__':
    nlp = English()
    tokenizer0 = nlp.Defaults.create_tokenizer(nlp)
    #tokenizer = Tokenizer.with_tokenizer(tokenizer0)
    #tokenizer = KBTokenizer('spacy')
    tokenizer = KBTokenizer('bert-multi')
    #tokenizer = KBTokenizer('sktkobert')
    #tokenizer = KBTokenizer('kbalbert')
    str1 = '난 학교에 간다.'
    seq1 = tokenizer.tokenize(str1)
    print(seq1)
    special_chars = [PAD_TAG, UNK_TAG]
    chars1 = '[PAD]fggg[PAD][UNK]sdfklsdf[PAD]'
    seq2 = tokenizer.tok2chars(chars1, special_chars)
    print(seq2)
    chars2 = 'fggg#$%[UNK][UNK]'
    seq2 = tokenizer.tok2chars(chars2, special_chars)
    print(seq2)
    temp = tokenizer._index_repetitive(chars1, '[PAD]')
    print(temp)
    str3 = 'I love you. [PAD] [PAD]'
    seq2 = tokenizer.tokenize(str3, flag_char=True)
    print(seq2)
    
    """ OUT
    ['I', 'love', 'you', '.']
    ['<PAD>', 'f', 'g', 'g', 'g', '<PAD>', '<UNK>', 's', 'd', 'f', 'k', 'l', 's', 'd', 'f', '<PAD>']
    ['f', 'g', 'g', 'g', '#', '$', '%', '<UNK>', '<UNK>']
    """
