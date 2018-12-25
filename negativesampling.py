import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collections import Counter, defaultdict


class Corpus:
    def __init__(self, wordlist, window_size):
        self.id2word = set(wordlist)
        self.word2id = {w: id for id, w in enumerate(set(wordlist))}
        self.wids = [self.word2id[w] for w in wordlist]
        self.contexts = self.make_contexts(window_size)

    def __len__(self):
        return len(self.word2id)

    def make_contexts(self, window_size):
        center = self.wids[window_size: -window_size]
        contexts = []
        for idx in range(window_size, len(self.wids)-window_size):
            cs = []
            cs.append(self.wids[idx])
            for t in range(-window_size, window_size+1):
                if t == 0:
                    continue
                cs.append(self.wids[idx+t])
            contexts.append(cs)
        return contexts


class Sampler:
    '''対象の単語に対するネガティブサンプリングされた単語を出力'''
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = len(corpus)
        self.counts = Counter(corpus.wids)
        self.word_p = np.power(list(self.counts.values()), power)
        self.word_p /= np.sum(self.word_p)

    def get_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_index = target[i]
            p[target_index] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        negative_sample = self.convert(negative_sample)
        return negative_sample

    def convert(self, batch):
        return torch.tensor(batch, dtype=torch.long)


def test():
    data = open('./data/data.txt', 'r').read().split()
    corpus = Corpus(data, 2)
    context = corpus.contexts
    print(f'corpus_size: {len(corpus.wids)}')
    print(f'vocab_size: {len(corpus.id2word)}')
    print(f'context_sample: {corpus.contexts[0]}')

    sampler = Sampler(corpus, 0.75, 5)
    for c in np.array([[1, 2, 3, 4, 5]]):
        print(sampler.get_sample(c))


if __name__ == '__main__':
    test()