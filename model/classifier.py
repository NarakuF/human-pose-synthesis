import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.process_text import tokenizer
from utils.utils import create_emb_layer
from pose_dataset import PoseDataset
import matplotlib.pyplot as plt
import numpy as np
from numpy import random


class ActivityClassifier(nn.Module):
    def __init__(self, input_embeddings, output_size):
        super(ActivityClassifier, self).__init__()
        self.annotate_embed_size = 128  # output encoded annotation size
        self.emb_layer = create_emb_layer(input_embeddings, non_trainable=True)
        self.rnn = nn.LSTM(input_size=input_embeddings.size()[1],
                           hidden_size=self.annotate_embed_size,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)
        self.main = nn.Sequential(
            nn.Linear(self.annotate_embed_size * 2, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, output_size)
        )

    def forward(self, annotate):
        embed_annotate = self.emb_layer(annotate)
        x, hidden = self.rnn(embed_annotate)
        encoded_x = x[:, 0, :] + x[:, -1, :]
        return self.main(encoded_x)


def anno2padded(anno, word2idx, max_len=15):
    tokens = tokenizer(anno)
    tokens = [word2idx.get(t, 0) for t in tokens]
    while len(tokens) < max_len:
        tokens.append(0)
    padded = torch.tensor([tokens], dtype=torch.long).cuda()
    return padded


def normalize_res(res):
    normalized = torch.squeeze(res.detach().cpu())
    normalized = normalized.numpy()
    normalized = np.exp(normalized)
    normalized = normalized / np.sum(normalized)
    return normalized


def get_label(probs):
    rand_prob = random.random()
    prob = 0.0
    for i, p in enumerate(probs):
        prob += p
        if rand_prob < prob:
            return i
    return -1


if __name__ == "__main__":
    dataset = PoseDataset('../data/data_list_50.csv', '../data', text_only=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    '''embeddings = dataset.embeddings
    num_class = len(set(dataset.data_list['label']))
    model = ActivityClassifier(embeddings, num_class)

    # Settings:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    num_epochs = 50

    model = model.to(device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        model.train()
        for i, sample in enumerate(dataloader):
            x_sample = sample['annotate'].to(device)
            y_sample = sample['label'].to(device)
            optimizer.zero_grad()
            y_output = model(x_sample)
            loss = criterion(y_output, y_sample)
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print('[epoch %d][idx %d]\tLoss: %.4f'
                      % (epoch + 1, i + 1, loss.item()))

    torch.save(model, './classifier_50.pth')'''

    model = torch.load('./classifier_50.pth')

    '''model.eval()
    correct = 0
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            x_sample = sample['annotate'].cuda()
            y_sample = sample['label']
            y_output = model(x_sample)
            y_max, idx = torch.max(y_output.cpu(), 1)
            correct += (idx == y_sample).sum().item()

    print(correct)
    print(correct / len(dataset))'''

    s = 'golf'
    s = anno2padded(s, dataset)
    with torch.no_grad():
        y = model(s)
    res = normalize_res(y)

    # plt.bar(np.arange(50), np.sort(res)[::-1])
    # plt.show()
    print(np.sum(res))
    print(np.max(res))
    print(np.min(res))
    print(np.mean(res))

    label = get_label(res)
    print(label)
