import torch.nn as nn

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
