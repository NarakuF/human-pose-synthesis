import os
import cv2
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import models
import torch.optim as optim

from skimage import io, transform
from utils.process_img import Rescale, DynamicCrop, ToTensor
from utils.process_text import tokenizer, get_embeddings, get_word2idx

TEXT_FEATURE = './intermediate/text_feature.pk'


class PoseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, brand_new=False):
        self.data_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Parsing text:
        if brand_new or not os.path.exists(TEXT_FEATURE):
            annotation_list = list(self.data_list['annotate'])
            self.word2idx, self.annotations, wv = get_word2idx(annotation_list)
            self.embeddings = get_embeddings(self.word2idx, wv)
            self.padded_annotate = nn.utils.rnn.pad_sequence(
                [torch.tensor([self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in
                               tokens]) for tokens in self.annotations])
            save_file = {'word2idx': self.word2idx, 'embeddings': self.embeddings,
                         'padded_annotate': self.padded_annotate}
            with open(TEXT_FEATURE, 'wb') as f:
                pickle.dump(save_file, f)
        else:
            with open(TEXT_FEATURE, 'rb') as f:
                save_file = pickle.load(f)
            self.word2idx = save_file['word2idx']
            self.embeddings = save_file['embeddings']
            self.padded_annotate = save_file['padded_annotate']

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        data_instance = self.data_list.iloc[idx]
        raw = cv2.imread(data_instance['raw'])
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        parsing = cv2.imread(data_instance['parsing'])
        pose = cv2.imread(data_instance['pose'])
        annotate = self.padded_annotate[:, idx]
        sample = {'raw': raw, 'parsing': parsing, 'pose': pose, 'annotate': annotate}
        if self.transform:
            sample = self.transform(sample)
        return sample


def print_sample(sample):
    print(sample['annotate'])
    fig = plt.figure(figsize=(10, 10))
    for i in range(3):
        fig.add_subplot(1, 3, i + 1)
        img = sample[list(sample.keys())[i]]
        plt.imshow(img)
    plt.show()
    return

# How to use:
# composed = transforms.Compose([Rescale(512),
#                                DynamicCrop(50),
#                                Rescale((350, 256))])

# pose_dataset = PoseDataset('./data/data_list.csv', './data', composed)
# pose_dataloader = DataLoader(pose_dataset, batch_size=16, shuffle=True, num_workers=4)
