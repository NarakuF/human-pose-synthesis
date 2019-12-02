import string
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import torch
import numpy as np
from utils.utils import WORD2VEC_PATH


def tokenizer(doc):
    return [t.lower() for t in word_tokenize(doc) if t not in string.punctuation]


def get_embeddings(word2idx, wv):
    embeddings = np.zeros((len(word2idx) + 1, 300))
    for word, i in word2idx.items():
        if word == '<pad>':
            pass
        elif word == '<unk>':
            embeddings[i] = wv['unk']
        else:
            embeddings[i] = wv[word]
    embeddings = torch.tensor(embeddings)
    return embeddings


def get_word2idx(annotation_list):
    wv = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
    word2idx = {'<pad>': 0}
    annotations = []
    for line in annotation_list:
        tokens = tokenizer(line)
        annotations.append(tokens)
        for t in tokens:
            if t in wv and t not in word2idx:
                word2idx[t] = len(word2idx)
    word2idx['<unk>'] = len(word2idx)
    return word2idx, annotations, wv
