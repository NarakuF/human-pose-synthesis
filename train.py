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
import torchvision.utils as vutils

from utils.process_img import Rescale, DynamicCrop, ToTensor, CenterCrop
from utils.func import weights_init, random_annotate, gaussian_noise
from pose_dataset import PoseDataset, print_sample
from model.generator import Generator
from model.discriminator import Discriminator
from utils.process_text import tokenizer, get_embeddings, get_word2idx


import pytorch_ssim
import pickle


opt = {'b1': 0.5, 
       'b2': 0.999, 
       'batch_size': 64, 
       'channels': 1, 
       'img_size': 32, 
       'latent_dim': 100, 
       'lr': 0.0002, 
       'n_classes': 200, 
       'n_cpu': 8, 
       'n_epochs': 1, 
       'sample_interval': 400}
img_size = 64
batch_size = 2
img_shape = (3, 64, 64)
composed = transforms.Compose([Rescale(512),
                               DynamicCrop(30),
                               Rescale((img_size, img_size))])

pose_dataset = PoseDataset('./data/data_list_label.csv', './data', transform = composed, gray_scale = False)
pose_dataloader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


adversarial_loss = torch.nn.MSELoss()
consistency_loss = pytorch_ssim.SSIM()#losses.SSIM(window_size = 5, reduction = 'sum')

generator_parsing = Generator().cuda()
discriminator_parsing = Discriminator().cuda()
generator_pose= Generator().cuda()
discriminator_pose = Discriminator().cuda()



optimizer_G_parsing = torch.optim.Adam(generator_parsing.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
optimizer_D_parsing = torch.optim.Adam(discriminator_parsing.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
optimizer_G_pose = torch.optim.Adam(generator_pose.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))
optimizer_D_pose = torch.optim.Adam(discriminator_pose.parameters(), lr=opt['lr'], betas=(opt['b1'], opt['b2']))



test_batch_size = 100
torch.manual_seed(4995)
fixed_z = torch.randn(test_batch_size, opt['latent_dim']).cuda()
fixed_gen_labels = torch.from_numpy(np.arange(test_batch_size)).cuda()

gen_img_list_pose, gen_img_list_parsing = [], []
G_loss_pose, D_loss_pose, G_loss_parsing, D_loss_parsing = [], [], [], []

# Train
n_epochs = 5
for epoch in range(n_epochs):
    for i, sample in enumerate(pose_dataloader):
        parsing = sample['parsing'].cuda()
        pose = sample['pose'].cuda()
        labels = sample['label'].cuda()
        
        b_size = parsing.shape[0]
        # Adversarial ground truths
        valid = torch.full((b_size, ), 1.0).cuda()
        fake = torch.full((b_size, ), 0.0).cuda()
        
        # -----------------
        #  Train Generator
        # -----------------
        

        # Sample noise and labels as generator input
        z = torch.randn(b_size, opt['latent_dim']).cuda()
        gen_labels = torch.from_numpy(np.random.randint(0, 200, b_size)).cuda()
        
        # Pose
        optimizer_G_pose.zero_grad()
        gen_parsing = generator_parsing(z, gen_labels)
        gen_pose = generator_pose(z, gen_labels)
        loss_consistency = consistency_loss(gen_parsing, gen_pose)    
        validity_pose = discriminator_pose(gen_pose, gen_labels)
        g_loss_pose = adversarial_loss(validity_pose.view(-1), valid)
        g_loss_pose = (g_loss_pose + loss_consistency) / 2
        g_loss_pose.backward()
        optimizer_G_pose.step()
        
        # Parsing
        optimizer_G_parsing.zero_grad()
        gen_parsing = generator_parsing(z, gen_labels)
        gen_pose = generator_pose(z, gen_labels)
        loss_consistency = consistency_loss(gen_parsing, gen_pose)
        validity_parsing = discriminator_parsing(gen_parsing, gen_labels)
        g_loss_parsing = adversarial_loss(validity_parsing.view(-1), valid)
        g_loss_parsing = (g_loss_parsing + loss_consistency) / 2
        g_loss_parsing.backward()
        optimizer_G_parsing.step()
        
        G_loss_parsing.append(g_loss_parsing.item())
        G_loss_pose.append(g_loss_pose.item())
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D_parsing.zero_grad()
        optimizer_D_pose.zero_grad()

        # Loss for real images
        
        validity_real = discriminator_pose(pose, labels)
        d_real_loss_pose = adversarial_loss(validity_real.view(-1), valid)
        validity_real = discriminator_parsing(parsing, labels)
        d_real_loss_parsing = adversarial_loss(validity_real.view(-1), valid)
        
        # Loss for fake images
        validity_fake = discriminator_pose(gen_pose.detach(), gen_labels)
        d_fake_loss_pose = adversarial_loss(validity_fake.view(-1), fake)
        validity_fake = discriminator_parsing(gen_parsing.detach(), gen_labels)
        d_fake_loss_parsing = adversarial_loss(validity_fake.view(-1), fake)

        # Total discriminator loss
        d_loss_pose = (d_real_loss_pose + d_fake_loss_pose) / 2
        d_loss_parsing = (d_real_loss_parsing + d_fake_loss_parsing) / 2
        D_loss_pose.append(d_loss_pose.item())
        D_loss_parsing.append(d_loss_parsing.item())
        
        d_loss_pose.backward()
        d_loss_parsing.backward()
        
        optimizer_D_parsing.step()
        optimizer_D_pose.step()

        if i and i % 5 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D_pose loss: %f] [G_pose loss: %f] [D_parsing loss: %f] [G_parsing loss: %f]"
                % (epoch, n_epochs, i, len(pose_dataloader), d_loss_pose.item(), g_loss_pose.item(), d_loss_parsing.item(), g_loss_parsing.item())
            )
        
        if i and i % 30 == 0:
            gen_parsing = generator_parsing(fixed_z, fixed_gen_labels)
            gen_pose = generator_pose(fixed_z, fixed_gen_labels)
            gen_img_list_parsing.append(gen_parsing)
            gen_img_list_pose.append(gen_pose)
            
            img_parsing = np.reshape(gen_parsing[0].cpu().detach().numpy(), (64, 64, 3))
            img_pose = np.reshape(gen_pose[0].cpu().detach().numpy(), (64, 64, 3))
            
            fig=plt.figure(figsize=(8, 8))
            fig.add_subplot(1, 2, 1)
            plt.imshow(img_parsing)
            fig.add_subplot(1, 2, 2)
            plt.imshow(img_pose)
            plt.show()


with open('gen_parsing_uni.pk', 'wb') as f:  
    pickle.dump(gen_img_list_parsing, f)

with open('gen_pose_uni.pk', 'wb') as f:
    pickle.dump(gen_img_list_pose, f)
    
loss = {'G_loss_pose': G_loss_pose, 'D_loss_pose': D_loss_pose, 'G_loss_parsing': G_loss_parsing, 'D_loss_parsing': D_loss_parsing}
with open('loss_uni.pk', 'wb') as f:
    pickle.dump(loss, f)