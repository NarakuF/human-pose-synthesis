import os
import json
import pathlib


def select_action(action=''):
    res = []
    annotation_file = './image2annotation.json'
    with open(annotation_file) as f:
        annotations = json.load(f)
    if not action:
        return res
    for img_id, anno in annotations.items():
        if action in anno:
            res.append(img_id)
    return res


data_dir = pathlib.Path('./MPII_sample1_json')

valid_id = []
with os.scandir(data_dir) as d:
    for file in d:
        with open(file.path) as f:
            keypoints = json.load(f)
            people = keypoints.get('people', None)
            if people and len(people) == 1:
                name = file.name.split('_')[0]
                valid_id.append(name + '.jpg')

print('Total number of valid id: ', len(valid_id))

with open('./valid_id.txt', 'w+') as f:
    for i in valid_id:
        f.write(i + '\n')
