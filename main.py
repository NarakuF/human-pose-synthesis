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

from utils.process_img import Rescale, DynamicCrop, ToTensor, CenterCrop
from utils.func import weights_init
from pose_dataset import PoseDataset, print_sample
from model.generator import PoseGenerator
from model.discriminator import PoseDiscriminator


if __name__ == "__main__":
    composed = transforms.Compose([Rescale(512),
                                   DynamicCrop(30),
                                   Rescale((128, 128))])

    pose_dataset = PoseDataset('./data/sample_data_list.csv', './data', transform=composed)
    pose_dataloader = DataLoader(pose_dataset, batch_size=10, shuffle=True, num_workers=4)

    embeddings = pose_dataset.embeddings

    # Generator
    netG = PoseGenerator(embeddings).cuda()
    netG.apply(weights_init)
    # Discriminator
    netD = PoseDiscriminator(embeddings).cuda()
    netD.apply(weights_init)

    # Settings:
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.Adam(netG.parameters())

    # Training Loop
    real_label = 1
    fake_label = 0

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    num_epochs = 1

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, sample in enumerate(pose_dataloader):
            batch_size = sample['raw'].shape[0]
            # reformat the shape to be (batch_size, 3, 128, 128)
            real_pose = torch.reshape(sample['raw'], (batch_size, 3, 128, 128)).float().cuda()

            annotate = sample['annotate'].cuda()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            label = torch.full((batch_size,), real_label).cuda()
            output = netD(real_pose, annotate).view(-1)
            errD_real = criterion(output, label)  # Calculate loss on all-real batch
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            noise = torch.randn(batch_size, 64, 1, 1).cuda()
            fake_pose = netG(noise, annotate)
            label.fill_(fake_label)
            output = netD(fake_pose.detach(), annotate).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()  # Update D

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            output = netD(fake_pose, annotate).view(
                -1)  # Since we just updated D, perform another forward pass of all-fake batch through D
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(pose_dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            #         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            #             with torch.no_grad():
            #                 fake = netG(fixed_noise).detach().cpu()
            #             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    torch.cuda.empty_cache()

    fake_img = None
    for i, sample in enumerate(pose_dataloader):
        batch_size = sample['raw'].shape[0]
        annotate = sample['annotate'].cuda()
        noise = torch.randn(batch_size, 64, 1, 1).cuda()
        fake_img = netG(noise, annotate)
        print(fake_img.shape)
        break

    img = torch.reshape(fake_img[0], (128, 128, 3)).cpu().detach().numpy()
    plt.imshow(img)
    plt.show()

    print(pose_dataset[1]['raw'].shape)
