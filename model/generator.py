import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create_emb_layer(embeddings, non_trainable=False):
    num_embeddings, embedding_dim = embeddings.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': embeddings})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


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
        #print(encoded_x.shape)
        encoded_annotate = torch.reshape(encoded_x, (-1, encoded_x.shape[1], 1, 1))
        #print(annotate.view(-1, annotate.shape[1]))
        #print(encoded_annotate.view(-1, encoded_x.shape[1]))
        # Generator的input: concatenate noise + annotation
        input_x = torch.cat((encoded_annotate, noise), 1)
        return self.main(input_x)


img_shape = (3, 128, 128)
class PoseGeneratorL(nn.Module):
    def __init__(self, embeddings):
        super(PoseGeneratorL, self).__init__()
        self.noise_size = 64
        self.hidden_size = 0 #64
        self.emb_layer = create_emb_layer(embeddings, non_trainable=True)
        # self.rnn = nn.GRU(input_size = embeddings.size()[1], 
        #                   hidden_size = self.hidden_size, 
        #                   num_layers = 2,
        #                   batch_first = True)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.noise_size + self.hidden_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, annotate):
        # batch_size = annotate.shape[0]
        # noise = noise.view(batch_size, -1)

        # embed_annotate = self.emb_layer(annotate)
        # # 找到最后一个non-zero的idx
        # idx = torch.min(embed_annotate==0, dim=1).indices
        # x, hidden = self.rnn(embed_annotate)
        # #encoded_annotate = x[torch.arange(x.size(0)), idx] 
        # encoded_annotate = x[:,-1,:]

        # # Concatenate annotate embedding and image to produce input
        # gen_input = torch.cat((encoded_annotate, noise), 1)

        # img = self.model(gen_input)
        # img = img.view(img.size(0), *img_shape)
        # return img

        batch_size = 128
        noise = noise.view(batch_size, -1)
        img = self.model(noise)
        img = img.view(img.size(0), *img_shape)


        return img



class Generator(nn.Module):
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