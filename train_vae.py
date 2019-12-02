import cv2
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
from torch.autograd import Variable

from utils.process_img import Rescale, DynamicCrop, ToTensor, CenterCrop
from utils.utils import weights_init
from pose_dataset import PoseDataset, print_sample
from model.generator import PoseGeneratorDC, PoseGeneratorL, Generator
from model.discriminator import PoseDiscriminatorDC, Discriminator
from utils.process_text import tokenizer, get_embeddings, get_word2idx

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from model.vae import *


config = {'batch_size': 32,
          'num_epochs': 100,
          'num_samples': 64,
          'num_every_nth_epoch': 1,
          'num_gpus': 1,
          'num_workers': 0,
          'ckpt': False,
          'ckpt_path': 'ckpt',
          'data_path': 'data',
          'img_path': 'img',
          'output_path': '',
          'dataset': 'mnist',
          'training_digits': 0,
          'z_dim': 256,
          'lr_adam': 0.001,
          'beta_1': 0.5,
          'beta_2': 0.999,
          'std': 0.02,
          'num_filters_in_final_layer': 128,
          'num_conv_layers': 4,
          'model': 'vae',
          'img_size': 64,
          'num_channels': 1,
          'c_dim': [128, 4, 4]}

img_size = 64
batch_size = 4
num_epoch = 20
target = 'pose' # or 'parsing'

composed = transforms.Compose([Rescale(512),
                               DynamicCrop(30),
                               Rescale((img_size, img_size))])

pose_dataset = PoseDataset('./data/truncate_data_list.csv', './data', transform = composed, gray_scale = True)
pose_dataloader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


v = VAE(config).cuda()


bce_loss = torch.nn.BCELoss(size_average=False)
v_optim = torch.optim.Adam(v.parameters(), lr=config['lr_adam'], betas=(config['beta_1'], config['beta_2']))


# Training
for epoch in range(num_epoch):
    running_loss = []
    for i, sample in enumerate(pose_dataloader, 0):
        images = sample[target]
        batch_size = images.shape[0]
        images = torch.reshape(images, (-1, 1, 64, 64))

        x = Variable(images.type(torch.cuda.FloatTensor))
        x_r = v(x)

        v.zero_grad()
        loss_r = bce_loss(x_r, x) / batch_size
        loss_kl = torch.mean(.5 * torch.sum((v.mu**2) + torch.exp(v.log_sigma_sq) - 1 - v.log_sigma_sq, 1))
        loss = loss_r + loss_kl
        loss.backward()
        v_optim.step()
        running_loss.append(loss.item())

    print("Epoch", epoch, "- Running Loss:", np.mean(running_loss))


# Visualize:
# for i in range(10,20):
#     sample = pose_dataset[i]
#     images = torch.reshape(torch.from_numpy(sample['parsing']), (1, 1, 64, 64))

#     x = Variable(images.type(torch.cuda.FloatTensor))
#     x_r = v(x) # reconstructed x

#     img = torch.reshape(x, (64, 64)).cpu().detach().numpy()
#     img_r = torch.reshape(x_r, (64, 64)).cpu().detach().numpy()

#     fig = plt.figure(figsize=(5,5))
#     fig.add_subplot(1,2,1)
#     plt.imshow(img)
#     fig.add_subplot(1,2,2)
#     plt.imshow(img_r)
#     plt.show()


torch.save(v, './intermediate/vae_' + target + '.pth')