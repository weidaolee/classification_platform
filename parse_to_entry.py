import glob
import tqdm
from functools import partial
import pandas as pd

entry = pd.read_csv('./data/Data_Entry_2017.csv')
img_paths = glob.glob('./data/images_*/images/*.png')
train_list = pd.read_csv('./data/train_val_list.txt', header=None)[0].tolist()


def train_test_split(x):
    for s in train_list:
        if x in s:
            train_list.remove(s)
            return 'train'
        else:
            return 'test'


def parse_to_path(x):
    for p in img_paths:
        if x in p:
            img_paths.remove(p)
            return p


def get_all_tags():
    all_tags = []
    for s in finding_labels:
        s = s.split('|')
        for _s in s:
            if _s not in all_tags:
                all_tags.append(_s)

    all_tags = sorted(all_tags)

    all_tags.remove('No Finding')

    return all_tags


entry['Train or Test'] = entry['Image Index'].apply(
    lambda x: train_test_split(x))
entry['Image Index'] = entry['Image Index'].apply(lambda x: parse_to_path(x))

finding_labels = entry['Finding Labels'].tolist()

all_tags = get_all_tags()


def _expand(x, t):
    if t in x:
        return 1
    else:
        return 0
    
for t in tqdm.tqdm(all_tags):
    expand = partial(_expand, t=t)
    entry[t] = entry['Finding Labels'].apply(lambda x: expand(x))

stats = entry[[
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
]].mean()

entry.drop(['Finding Labels'], axis=1, inplace=True)

train_valid_entry = entry[entry['Train or Test'] == 'train']
test_entry = entry[entry['Train or Test'] != 'train']

train_valid_entry.to_csv('train_valid_entry.csv')
test_entry.to_csv('test_entry.csv')
