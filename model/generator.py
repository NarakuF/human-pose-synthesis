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

    
ngf = 64
class Generator(nn.Module):
    def __init__(self, embeddings):
        super(Generator, self).__init__()
        self.emb_layer = create_emb_layer(embeddings, non_trainable=True)
        self.rnn = nn.GRU(input_size = embeddings.size()[1], 
                          hidden_size = 64, 
                          num_layers = 2,
                          batch_first = True)
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(64, ngf * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, data):
        x = data['annotate']
        x = self.emb_layer(x)
        x, hidden = self.rnn(x)
        decoded_x = torch.reshape(x[:,-1,:], (-1, x[:,-1,:].shape[1], 1, 1))
        return self.main(decoded_x)