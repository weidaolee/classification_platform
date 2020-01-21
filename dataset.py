import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NIHDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None, mode='valid'):

        self.data = pd.read_csv(path)
        self.n_images = self.__len__()
        self.labels = np.asarray(self.data.iloc[:, -14:])
        self.mode = mode

    def __getitem__(self, index):
        label_as_tensor = self.labels[index]
        img_path = self.data.iloc[index]['Image Index'].replace('.png', '.npz')

        img = np.load(img_path)['image']
        if self.mode == 'train':
            img = self.lr_flip(img)

        # Transform image to tensor
        img = torch.from_numpy(img)
        # Return image and the label

        # return (img_as_tensor.unsqueeze(2).repeat(1, 1, 3), label_as_tensor)
        return (img_path, img, label_as_tensor)

    def lr_flip(self, img, p=0.5):
        if np.random.uniform() < p:
            img = img[..., ::-1] - np.zeros_like(img)
        return img

    def __len__(self):
        return len(self.data.index)
