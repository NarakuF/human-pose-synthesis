import os
import json
import pandas as pd

source_path = '.'
RAW = '/MPII_sample1'
PARSING = '/MPII_sample1_parsing'
POSE = '/MPII_sample1_pose'


# Loading data:
with open(source_path+'/image2annotation.json', 'r') as f:
    image2annotation = json.load(f)

valid_id = []
with open(source_path+'/valid_id.txt', 'r') as f:
    for line in f:
        valid_id.append(line.strip())
valid_id = set(valid_id)

img2path = {'raw':[], 'parsing':[], 'pose':[], 'annotate':[]}
for f in os.listdir(source_path + RAW):
    if f not in valid_id:
        continue
        
    img_id = f.split('.')[0]
    raw_path = './data'+RAW+'/'+f
    parsing_path = './data'+PARSING+'/'+img_id+'_vis.png'
    pose_path = './data'+POSE+'/'+img_id+'_rendered.png'

    img2path['raw'].append(raw_path)
    img2path['parsing'].append(parsing_path)
    img2path['pose'].append(pose_path)
    img2path['annotate'].append(image2annotation[f])

path_list = pd.DataFrame.from_dict(img2path)
path_list.to_csv(source_path+'/data_list.csv', index = False)