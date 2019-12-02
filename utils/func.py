import numpy as np

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def random_annotate(batch_size, dataset, random_state=None):
    if not random_state:
        np.random.seed(random_state)
    idx = np.random.randint(0, dataset.padded_annotate.shape[1], batch_size)
    return torch.from_numpy(np.transpose(dataset.padded_annotate.numpy()[:, idx]))


def gaussian_noise(inputs, mean=0, sd=0.01):
    input_array = inputs.cpu().data.numpy()
    noise = np.random.normal(loc=mean, scale=sd, size=np.shape(input_array))
    out = np.add(input_array, noise)

    output_tensor = torch.from_numpy(out)
    out_tensor = torch.tensor(output_tensor)
    out = out_tensor.cuda()
    out = out.float()
    return out
