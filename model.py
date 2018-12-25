import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, Dataset

from negativesampling import Corpus, Sampler

from collections import Counter, defaultdict
from tqdm import tqdm


class SkipGram(nn.Module):
    def __init__(self, num_classes, embed_size, sampler):
        super().__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.sampler = sampler

        self.out_embed = nn.Embedding(num_classes, embed_size, sparse=True)
        self.out_embed.weight.data.uniform_(-0.1, 0.1)
        self.in_embed = nn.Embedding(num_classes, embed_size, sparse=True)
        self.in_embed.weight.data.uniform_(-0.0, 0.0)

    def forward(self, input_labels, out_labels, num_sampled):
        batch_size = len(input_labels)
        input = self.in_embed(input_labels)
        output = self.out_embed(out_labels)
        batch_size, n_context, n_units = output.shape
        log_target = (input.unsqueeze(1) * output).sum(1).squeeze().sigmoid().log().sum(1)

        noise = self.sampler.get_sample(out_labels)
        noise = self.out_embed(noise)
        sum_log_sampled = torch.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


def test():
    data = open('./data/data.txt', 'r').read().split()
    corpus = Corpus(data[:10000], 2)
    context = corpus.contexts
    sampler = Sampler(corpus, 0.75, 5)

    context = torch.tensor(context, dtype=torch.long)
    loader = DataLoader(context, batch_size=100)

    model = SkipGram(len(corpus.id2word), 2, sampler)
    optimizer = optim.SGD(model.parameters(), 0.1)

    for c in tqdm(loader):
        input = c[:, 0]
        target = c[:, 1:]
        loss = model(input, target, 3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'loss: {loss.data}')


if __name__ == '__main__':
    test()