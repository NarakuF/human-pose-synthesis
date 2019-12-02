import os
import torch
import pickle
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils.process_text import get_embeddings, get_word2idx
from model.utils import create_emb_layer


class ActivityDataset(Dataset):
    def __init__(self, csv_file, text_feature='', brand_new=False):
        self.data_list = pd.read_csv(csv_file)
        self.activities = list(self.data_list['annotate'])
        self.labels = list(self.data_list['label'])
        self.num_class = len(set(self.labels))
        classes = list(set(self.activities))
        print(classes)
        self.y = torch.zeros(len(self.activities), dtype=torch.long)
        for idx in range(len(self.activities)):
            a = self.activities[idx]
            self.y[idx] = classes.index(a)
        # Parsing text:
        if brand_new or not os.path.exists(text_feature):
            annotation_list = list(self.data_list['annotate'])
            self.word2idx, self.annotations, wv = get_word2idx(annotation_list)
            self.embeddings = get_embeddings(self.word2idx, wv)
            self.padded_annotate = nn.utils.rnn.pad_sequence(
                [torch.tensor([self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in
                               tokens]) for tokens in self.annotations])
            save_file = {'annotations': self.annotations, 'word2idx': self.word2idx, 'embeddings': self.embeddings,
                         'padded_annotate': self.padded_annotate}
            with open(text_feature, 'wb+') as f:
                pickle.dump(save_file, f)
        else:
            with open(text_feature, 'rb') as f:
                save_file = pickle.load(f)
            self.annotations = save_file['annotations']
            self.word2idx = save_file['word2idx']
            self.embeddings = save_file['embeddings']
            self.padded_annotate = save_file['padded_annotate']

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        annotate = self.padded_annotate[:, idx]
        item = {'annotate': annotate, 'class': self.y[idx], 'label': self.labels[idx]}
        return item


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


if __name__ == "__main__":
    dataset = ActivityDataset('../data/data_list_label.csv', '../intermediate/text_feature_label.pk')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    embeddings = dataset.embeddings
    num_class = dataset.num_class
    model = ActivityClassifier(embeddings, num_class)

    # Settings:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 20

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

    for i in range(10):
        s = 'golf'
        tokens = s.split(' ')
        tokens = [dataset.word2idx.get(t, 0) for t in tokens]
        max_len = dataset.padded_annotate.shape[0]
        while len(tokens) < max_len:
            tokens.append(0)
        s = torch.tensor([tokens], dtype=torch.long).to(device)
        y = model(s)
        max_y, label = torch.max(y, 1)
        print(label.item())
