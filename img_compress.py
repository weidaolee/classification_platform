import yaml
import argparse
import tqdm
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.transform import resize


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default='./config/defualt.cfg',
                        help='config file. see readme')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f)

    entry_path = [
        cfg['PATH']['TRAIN']['ENTRY_FILE'],
        cfg['PATH']['TEST']['ENTRY_FILE'],
    ]
    img_size = (cfg['IMGSIZE'], ) * 2

    def compress(p):
        img = plt.imread(p)
        if len(img) != 2:
            img = np.mean(img[..., :3], -1)
        img = resize(img, img_size)
        img = np.stack([img] * 3, 0)
        np.savez_compressed(p.replace('.png', '.npz'), image=img)

    for path in entry_path:
        img_paths = pd.read_csv(path)['Image Index'].tolist()

        with multiprocessing.Pool(20) as pool:
            gen = pool.imap(compress, img_paths)
            for _ in tqdm.tqdm(gen, total=len(img_paths)):
                pass
