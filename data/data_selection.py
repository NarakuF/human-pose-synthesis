import os
import json
import pathlib

data_dir = pathlib.Path('d:/Data/openpose/MPII1_json')

valid_id = []
with os.scandir(data_dir) as d:
    for file in d:
        with open(file.path) as f:
            keypoints = json.load(f)
            people = keypoints.get('people', None)
            if people and len(people) == 1:
                valid_id.append(file.name.split('_')[0])
print('Total number of valid id: ', len(valid_id))

with open('./valid_id.txt', 'w+') as f:
    for i in valid_id:
        f.write(i + '\n')
