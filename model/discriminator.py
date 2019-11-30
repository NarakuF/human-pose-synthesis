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


class PoseDiscriminatorDC(nn.Module):
    def __init__(self, embeddings):
        super(PoseDiscriminatorDC, self).__init__()
        self.annotate_embed_size = 64   # output encoded annotation size
        self.output_size = 128          # output image representation size after going through CNN
        self.emb_layer = create_emb_layer(embeddings, non_trainable=True)
        self.rnn = nn.LSTM(input_size = embeddings.size()[1], 
                           hidden_size = self.annotate_embed_size, 
                           num_layers = 2,
                           batch_first = True)
        ndf = 64
        nc = 3
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 64, 4, 1, 0, bias=False)
        )

        self.fc_1 = nn.Linear(self.output_size, 128)
        self.norm_1 = nn.BatchNorm1d(128)
        self.fc_2 = nn.Linear(128, 64)
        self.norm_2 = nn.BatchNorm1d(64)
        self.fc_3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, annotate):
        batch_size = annotate.shape[0]

        embed_annotate = self.emb_layer(annotate)
        x, hidden = self.rnn(embed_annotate)
        encoded_annotate = x[:,-1,:]

        x = self.main(image)
        encoded_img = x.view(batch_size, -1)

        fc_input = torch.cat((encoded_img, encoded_annotate), 1)
        x = F.relu(self.norm_1(self.fc_1(fc_input)))
        x = F.relu(self.norm_2(self.fc_2(x)))
        x = self.fc_3(x)

        return x

        #print(x.shape)

        # batch_size = annotate.shape[0]
        # embed_annotate = self.emb_layer(annotate)
        # x, hidden = self.rnn(embed_annotate)
        
        # encoded_annotate = torch.reshape(x[:,-1,:], (-1, x[:,-1,:].shape[1], 1, 1))

        # encoded_img = self.main(image)
        
        # # *最后一层fully connected layer的input (concatenate encoded_annotate and encoded_img) 
        # #input_x = torch.cat((encoded_img, encoded_annotate), 1)
        # input_x = encoded_img
        # input_x = input_x.view(batch_size, -1)

        # x = F.relu(self.norm_1(self.fc_1(input_x)))
        # x = F.relu(self.norm_2(self.fc_2(x)))
        # x = self.sigmoid(self.fc_3(x))

        # #decision = self.sigmoid(self.fc(input_x))


        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)