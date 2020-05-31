import os

import numpy as np
import skimage.color
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from dataset.color_sampling import sample_color_hints

lab_mean = [0.5, 0.5, 0.5]  # Per Rich Zhang Colorization paper
lab_std = [0.5, 0.5, 0.5]


class UserGuidedVideoDataset(Dataset):
    def __init__(self, path, augmentation: bool):
        self.path = path

        self.files = []
        for filename in os.listdir(path):
            self.files.append(filename)

        self.normalize_lab = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(lab_mean, lab_std),
        ])

        transform_list = [transforms.ToPILImage()]
        if augmentation:
            transform_list += [
                transforms.RandomCrop((320, 320)),
                transforms.RandomHorizontalFlip(),
            ]
        self.augmentation = transforms.Compose(transform_list)

    def __getitem__(self, index):
        path = os.path.join(self.path, self.files[index])
        rgb = self.augmentation(io.imread(path))
        lab = skimage.color.rgb2lab(rgb).astype(np.float32)
        lab = self.normalize_lab(lab)
        L_channel = lab[[0]]
        ab_channels = lab[1:]

        ab_hint, ab_mask, _ = sample_color_hints(ab_channels)
        return L_channel, ab_channels, ab_hint, ab_mask

    def __len__(self):
        return len(self.files)
