import cv2
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.process_img import Rescale, DynamicCrop
from pose_dataset import PoseDataset
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans


if __name__ == '__main__':
    # Parameters:
    lr = 0.0002
    beta1 = 0.5
    img_size = 64
    z_size = 16
    batch_size = 16
    composed = transforms.Compose([Rescale(512),
                                   DynamicCrop(30),
                                   Rescale((img_size, img_size))])

    pose_dataset = PoseDataset('./data/data_list.csv', './data', transform=composed, gray_scale=True, label=False)
    pose_dataloader = DataLoader(pose_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(len(pose_dataset))

    # Clustering on original image
    '''X = []
    for i in tqdm(range(len(pose_dataset))):
        sample = pose_dataset[i]
        img = sample['pose']
        flatten_img = np.reshape(img, (img.shape[0] ** 2))
        X.append(flatten_img)
    X = np.array(X)'''

    # with open('./intermediate/flatten_pose.pk', 'wb+') as f:
    #     pickle.dump(X, f)

    with open('./intermediate/flatten_pose.pk', 'rb') as f:
        feature_0 = pickle.load(f)

    kmeans = KMeans(n_clusters=50, random_state=0, verbose=1, n_jobs=8)
    kmeans.fit(feature_0)
    df = pd.read_csv('./data/data_list.csv')
    df['label'] = kmeans.labels_
    df.to_csv('./data/data_list_50.csv', index=False)

    # Creating sample group:
    '''label2imgs = {}
    for i in tqdm(range(len(labels))):
        img = pose_dataset[i]['pose']
        label2imgs[labels[i]] = label2imgs.get(labels[i], []) + [img]
    
    
    class_id = 20
    imgs = label2imgs[class_id]
    imgs_t = torch.from_numpy(np.array(imgs))
    imgs_t = torch.reshape(imgs_t, [-1, 1, 64, 64])
    
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(imgs_t, padding=2, normalize=True).cpu(), (1, 2, 0)))
    # plt.savefig('pure_class.png', dpi=512)
    plt.show()
    
    annotates = [' '.join(a) for a in pose_dataset.annotations]
    annotate2count = {}
    for i in tqdm(range(len(labels))):
        if labels[i] == 2:
            annotate2count[annotates[i]] = annotate2count.get(annotates[i], 0) + 1
    sorted(annotate2count.items(), key=lambda x: x[1])[::-1]
    
    
    # plt.rcdefaults()
    plt.figure(figsize=(128, 128))
    fig, ax = plt.subplots()
    
    # Example data
    annos = list(annotate2count.keys())
    annos = [a[:20] + (len(a) > 20) * '...' for a in annos]
    y_pos = np.arange(len(annos))
    counts = list(annotate2count.values())
    
    ax.barh(y_pos, counts, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(annos, rotation=55, fontsize=4)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('count')
    ax.set_title('activity distribution/ pose class')
    
    plt.savefig('activity_distribution.png', dpi=512)
    plt.show()'''
