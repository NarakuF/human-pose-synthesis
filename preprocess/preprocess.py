import json
import random
import pandas as pd


def get_experiment_dataset():
    with open('./config.json') as f:
        config = json.load(f)
    df = pd.read_csv(config['original_file'], delim_whitespace=True, skiprows=1)

    with open(config['experiment_file'], 'w+') as f:
        header = ' '.join(list(df))
        header += '\n'
        f.write(header)

    random.seed(config['seed'])
    dataset = ['train', 'query', 'gallery']
    for phase in dataset:
        orig_dataset = df[df.evaluation_status == phase]
        idx = random.sample(range(len(orig_dataset.index)), config[phase])
        exp_dataset = orig_dataset.iloc[idx]
        exp_dataset.to_csv(config['experiment_file'], header=False, index=False, sep=' ', mode='a')


if __name__ == '__main__':
    get_experiment_dataset()
