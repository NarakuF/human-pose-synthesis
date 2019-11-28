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


class Discriminator(nn.Module):
    def __init__(self, embeddings):
        super(Discriminator, self).__init__()
        self.hidden_size = 64 # output encoded annotation size
        self.image_size = 256 # input image size before going through CNN
        self.output_size = 64 # output image representation size after going through CNN
        self.emb_layer = create_emb_layer(embeddings, non_trainable=True)
        self.rnn = nn.GRU(input_size = embeddings.size()[1], 
                          hidden_size = self.hidden_size, 
                          num_layers = 2,
                          batch_first = True)
        ndf = 128
        self.main = nn.Sequential(
            # input is (nc=3) x 64 x 64
            nn.Conv2d(3, ndf*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, self.output_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_size)
        )
        self.fc = nn.Linear(self.output_size+self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, data):
        annotate = data['annotate'].cuda()
        embed_annotate = self.emb_layer(annotate)
        x, hidden = self.rnn(embed_annotate)
        
        encoded_annotate = torch.reshape(x[:,-1,:], (-1, x[:,-1,:].shape[1], 1, 1))
        encoded_img = self.main(image)
        
        # *最后一层fully connected layer的input (concatenate encoded_annotate and encoded_img) 
        print(encoded_img.shape, encoded_annotate.shape)  
        input_x = torch.cat((encoded_img, encoded_annotate), 1)
        input_x = input_x.view(4, -1)

        decision = self.sigmoid(self.fc(input_x))
        return decision