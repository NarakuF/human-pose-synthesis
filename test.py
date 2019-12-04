from model.classifier import ActivityClassifier, anno2padded, normalize_res, get_label
from model.discriminator import Discriminator
from model.generator import Generator

import torch
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

opt = {'b1': 0.5, 
       'b2': 0.999, 
       'batch_size': 64, 
       'channels': 1, 
       'img_size': 64, 
       'latent_dim': 32, 
       'lr': 0.0002, 
       'n_classes': 200, 
       'n_cpu': 8, 
       'n_epochs': 1, 
       'sample_interval': 400}

mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Model:
print("Loading Model...")
classifier = torch.load('./intermediate/classifier_200.pth')
classifier.to(device = mydevice)
netG_pose = torch.load('./intermediate/pose_netG_uni.pth')
netG_pose.to(device = mydevice)
netG_parsing = torch.load('./intermediate/parsing_netG_uni.pth')
netG_parsing.to(device = mydevice)

with open('./intermediate/word2idx.pk', 'rb') as f:
    word2idx = pickle.load(f)



# User entered annotation:
annotate = input('Please enter an annotation (<16 words): ')
while len(annotate.split(' ')) > 15:
    annotate = input('[!] Max length exceeded. Please enter an annotation (<16 words): ')



# Retrieving the potential cluster label based on pose distribution
s = anno2padded(annotate, word2idx)
with torch.no_grad():
    res = classifier(s)
res = normalize_res(res)
label = get_label(res)



# Generating pose and parsing based on retrieved label
z = torch.randn(1, opt['latent_dim'], device = mydevice)
label = torch.tensor([label], device = mydevice) 

netG_parsing.eval()
netG_pose.eval()
with torch.no_grad():
    gen_parsing = netG_parsing(z, label)
    gen_pose = netG_pose(z, label)

img_parsing = np.reshape(gen_parsing[0].cpu().detach().numpy(), (64, 64, 3))
img_pose = np.reshape(gen_pose[0].cpu().detach().numpy(), (64, 64, 3))
plt.imshow(img_parsing)
plt.savefig('./output/sample_parsing.png')
plt.imshow(img_pose)
plt.savefig('./output/sample_pose.png')
print('Pose, Parsing Generated.')
