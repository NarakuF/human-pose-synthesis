import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def create_emb_layer(embeddings, non_trainable=False):
    num_embeddings, embedding_dim = embeddings.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': embeddings})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


class Generator(nn.Module):
    def __init__(self, embeddings):
        super(Generator, self).__init__()
        self.hidden_size = 64 # output encoded annotation size
        self.noise_size = 64 # 初始的image size
        self.emb_layer = create_emb_layer(embeddings, non_trainable=True)
        self.rnn = nn.GRU(input_size = embeddings.size()[1], 
                          hidden_size = self.hidden_size, 
                          num_layers = 2,
                          batch_first = True)

        ngf = 64 # number of feature for the image
        in_feature_size = self.hidden_size + self.noise_size
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_feature_size, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        
    def forward(self, noise, data):
        # Get the encoded annotation from RNN
        annotate = data['annotate'].cuda()
        embed_annotate = self.emb_layer(annotate)
        x, hidden = self.rnn(embed_annotate)
        encoded_annotate = torch.reshape(x[:,-1,:], (-1, x[:,-1,:].shape[1], 1, 1))
        # Generator的input: concatenate noise + annotation
        input_x = torch.cat((encoded_annotate, noise), 1)
        return self.main(input_x)