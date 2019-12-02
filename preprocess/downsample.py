from scipy.io import loadmat
import cv2
import os
import json
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

source_path = './images/'
full_path = './MPII_clean_img/'
sample2_path = './MPII_clean_img_1/'
sample3_path = './MPII_clean_img_2/'
anno_file = './MPII_anno/mpii_human_pose_v1_u12_1.mat'
a = loadmat(anno_file)

info = a['RELEASE'][0][0]
count = 0
act_pool = set()
image2annotation = {}

for i, act in enumerate(info[-2]):
    line = act[0]
    if len(line[1]):
        act_pool.update([line[1][0]])
        count += 1

        annotation = line[1][0]
        image = info[0][0][i][0][0][0][0][0]

        image2annotation[image] = annotation

images = list(image2annotation.keys())

for img_name in tqdm(images):
    img = cv2.imread(source_path + str(img_name))
    if not os.path.isfile(source_path + str(img_name)):
        del image2annotation[img_name]
        continue
    # full version:
    cv2.imwrite(full_path + str(img_name), img)
    # downsampled 2:
    res = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(sample2_path + str(img_name), res)
    # downsampled 3:
    res = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(sample3_path + str(img_name), res)

with open('image2annotation.json', 'w') as outfile:
    json.dump(image2annotation, outfile)
