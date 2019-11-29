import nltk
import string
import json
import numpy as np
from numpy import random
from utils import process_text


def generator(vocab, bigram_vocab, max_len=10):
    #vocab = vocab.union(bigram_vocab)
    vocab = bigram_vocab
    vocab = list(vocab)
    n = random.randint(1, max_len)
    res = []
    while n > 0:
        idx = random.randint(0, len(vocab))
        token = vocab[idx]
        res.append(vocab[idx])
        n -= len(token.split(' '))
    return ' '.join(res)


if __name__ == '__main__':
    anno_fn = './data/image2annotation.json'

    with open(anno_fn) as f:
        annotations = json.load(f)

    annotations = set(annotations.values())
    vocabulary = set()
    bigram_vocabulary = set()
    for anno in annotations:
        tokens = process_text.tokenizer(anno)
        vocabulary.update(tokens)
        # then build bigram vocabulary
        if len(tokens) < 2:
            continue
        for i in range(len(tokens) - 1):
            bigram = ' '.join(tokens[i:i + 2])
            bigram_vocabulary.add(bigram)

    for i in range(30):
        s = generator(vocabulary, bigram_vocabulary, 6)
        print(s)
