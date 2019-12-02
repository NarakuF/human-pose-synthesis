import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

img_shape = (3, 64, 64)
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

def create_emb_layer(embeddings, non_trainable=False):
    num_embeddings, embedding_dim = embeddings.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': embeddings})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt['n_classes'], opt['n_classes'])

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt['latent_dim'] + opt['n_classes'], 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)

        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class PoseGeneratorDC(nn.Module):
    def __init__(self, embeddings):
        super(PoseGeneratorDC, self).__init__()
        self.annotate_embed_size = 128 	# output encoded annotation size
        self.z_size = 16 				# 初始的noise size
        self.emb_layer = create_emb_layer(embeddings, non_trainable=True)
        self.rnn = nn.LSTM(input_size = embeddings.size()[1], 
                           hidden_size = self.annotate_embed_size, 
                           num_layers = 2,
                           batch_first = True,
                           bidirectional = True)

        ngf = 64						 # number of feature for the image
        nc = 3
        in_feature_size = self.annotate_embed_size*2 + self.z_size  # *2因为bidirectional

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( in_feature_size, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, noise, annotate):
        # Get the encoded annotation from RNN
        embed_annotate = self.emb_layer(annotate)
        x, hidden = self.rnn(embed_annotate)
        encoded_x = x[:,0,:]+x[:,-1,:]

        encoded_annotate = torch.reshape(encoded_x, (-1, encoded_x.shape[1], 1, 1))
        input_x = torch.cat((encoded_annotate, noise), 1)
        return self.main(input_x)


class GeneratorDC(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d( ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)